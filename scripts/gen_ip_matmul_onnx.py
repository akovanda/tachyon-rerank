#!/usr/bin/env python3
import argparse
import onnx
from onnx import helper, TensorProto

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a simple MatMul/Gemm ONNX model A@q")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--n", type=int, default=None, help="Static N (rows of A)")
    ap.add_argument("--d", type=int, default=None, help="Static D (cols of A)")
    ap.add_argument("--dynamic", action="store_true", help="Use symbolic dims N,D")
    ap.add_argument("--op", choices=["matmul", "gemm", "conv"], default="matmul",
                    help="Operator to use (default: matmul)")
    ap.add_argument("--q-row", action="store_true",
                    help="Use q shape [1, D] instead of [D, 1] (Gemm only)")
    ap.add_argument("--bake-a-bin", default=None,
                    help="Path to raw f32 little-endian A matrix to bake as weights (n*d floats)")
    args = ap.parse_args()

    if args.dynamic:
        n_dim = "N"
        d_dim = "D"
    else:
        if args.n is None or args.d is None:
            ap.error("--n and --d are required unless --dynamic is set")
        n_dim = args.n
        d_dim = args.d

    baked = args.bake_a_bin is not None
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [n_dim, d_dim])
    if args.q_row:
        q_shape = [1, d_dim]
        y_shape = [1, n_dim] if baked else [n_dim, 1]
    else:
        q_shape = [d_dim, 1]
        y_shape = [n_dim, 1]
    q = helper.make_tensor_value_info("q", TensorProto.FLOAT, q_shape)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, y_shape)

    if args.op == "matmul":
        if args.q_row:
            ap.error("--q-row is only supported with --op gemm")
        node = helper.make_node("MatMul", ["A", "q"], ["y"], name="matmul")
        graph = helper.make_graph([node], "ip_matmul", [A, q], [y])
    elif args.op == "gemm":
        # Gemm: Y = A * B + C. For QNN EP, beta=1.0 is required on HTP.
        transB = 1 if args.q_row else 0
        if args.dynamic:
            node = helper.make_node("Gemm", ["A", "q"], ["y"], name="gemm",
                                    alpha=1.0, beta=0.0, transA=0, transB=transB)
            graph = helper.make_graph([node], "ip_gemm", [A, q], [y])
        else:
            if args.bake_a_bin:
                if args.op != "gemm":
                    ap.error("--bake-a-bin requires --op gemm")
                if not args.q_row:
                    ap.error("--bake-a-bin requires --q-row (q shape [1,D])")
                import struct
                with open(args.bake_a_bin, "rb") as f:
                    raw = f.read()
                expected = int(n_dim) * int(d_dim)
                floats = list(struct.unpack("<" + "f" * expected, raw))
                # Store weights as [D, N] so Gemm uses transB=0 (KxN) for FC.
                w_vals = [0.0 for _ in range(int(n_dim) * int(d_dim))]
                for i in range(int(n_dim)):
                    for j in range(int(d_dim)):
                        w_vals[j * int(n_dim) + i] = floats[i * int(d_dim) + j]
                w_init = helper.make_tensor("W", TensorProto.FLOAT, [d_dim, n_dim], w_vals)
                c_vals = [0.0 for _ in range(int(n_dim))]
                c_init = helper.make_tensor("C", TensorProto.FLOAT, [1, n_dim], c_vals)
                node = helper.make_node("Gemm", ["q", "W", "C"], ["y"], name="gemm",
                                        alpha=1.0, beta=1.0, transA=0, transB=0)
                graph = helper.make_graph([node], "ip_gemm_baked", [q], [y], initializer=[w_init, c_init])
            else:
                # Add a zero bias C so beta=1.0 doesn't change outputs.
                c_vals = [0.0 for _ in range(int(n_dim))]
                c_init = helper.make_tensor("C", TensorProto.FLOAT, [n_dim, 1], c_vals)
                node = helper.make_node("Gemm", ["A", "q", "C"], ["y"], name="gemm",
                                        alpha=1.0, beta=1.0, transA=0, transB=transB)
                graph = helper.make_graph([node], "ip_gemm", [A, q], [y], initializer=[c_init])
    else:
        # Conv1x1 with baked weights. Requires --bake-a-bin and static dims.
        if not args.bake_a_bin:
            ap.error("--op conv requires --bake-a-bin")
        if args.dynamic:
            ap.error("--op conv requires static --n/--d")
        import struct
        with open(args.bake_a_bin, "rb") as f:
            raw = f.read()
        expected = int(n_dim) * int(d_dim)
        floats = list(struct.unpack("<" + "f" * expected, raw))
        # W: [out_channels=n, in_channels=d, 1, 1]
        w_vals = [0.0 for _ in range(int(n_dim) * int(d_dim))]
        for oc in range(int(n_dim)):
            for ic in range(int(d_dim)):
                w_vals[oc * int(d_dim) + ic] = floats[oc * int(d_dim) + ic]
        w_init = helper.make_tensor("W", TensorProto.FLOAT, [n_dim, d_dim, 1, 1], w_vals)
        b_vals = [0.0 for _ in range(int(n_dim))]
        b_init = helper.make_tensor("B", TensorProto.FLOAT, [n_dim], b_vals)
        q4 = helper.make_tensor_value_info("q", TensorProto.FLOAT, [1, d_dim, 1, 1])
        y4 = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, n_dim, 1, 1])
        node = helper.make_node("Conv", ["q", "W", "B"], ["y"], name="conv1x1",
                                pads=[0, 0, 0, 0], strides=[1, 1])
        graph = helper.make_graph([node], "ip_conv_baked", [q4], [y4], initializer=[w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    onnx.checker.check_model(model)
    onnx.save(model, args.out)
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
