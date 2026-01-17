import argparse
from pathlib import Path

import onnx
from onnx import helper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch a CLIP text ONNX model by replacing Range nodes with constants."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the text model ONNX file (e.g., clip_text.onnx).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output path for the patched model (default: clip_text_fixed.onnx beside input).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=77,
        help="Sequence length to bake into the Range constants (default: 77).",
    )
    parser.add_argument(
        "--batch-len",
        type=int,
        default=1,
        help="Batch length to bake for the Range_1 node (default: 1).",
    )
    return parser.parse_args()


def make_range_tensor(name: str, length: int) -> onnx.TensorProto:
    return helper.make_tensor(
        name=name,
        data_type=onnx.TensorProto.INT64,
        dims=[length],
        vals=list(range(length)),
    )


def patch_ranges(
    model: onnx.ModelProto, seq_len: int, batch_len: int
) -> tuple[onnx.ModelProto, int]:

    new_nodes = []
    replaced = 0
    for node in model.graph.node:
        if node.op_type == "Range" and node.output:
            length = seq_len
            if node.name.endswith("Range_1"):
                length = batch_len
            tensor = make_range_tensor("range_const", length)
            const = helper.make_node(
                "Constant",
                inputs=[],
                outputs=[node.output[0]],
                value=tensor,
                name=f"{node.name}_const" if node.name else "",
            )
            new_nodes.append(const)
            replaced += 1
        else:
            new_nodes.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    return model, replaced


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Model not found: {input_path}")
    output_path = args.output or input_path.with_name("clip_text_fixed.onnx")

    model = onnx.load(str(input_path))
    model, replaced = patch_ranges(model, args.seq_len, args.batch_len)
    onnx.save(model, str(output_path))
    print(f"patched {replaced} Range node(s) -> {output_path}")


if __name__ == "__main__":
    main()
