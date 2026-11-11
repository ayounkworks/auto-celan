from local_runner import build_parser


def test_parser_accepts_input_output_workers():
    args = build_parser().parse_args([
        r"D:\Manga\Chapter 01", "--output", r"D:\Out\Chapter 01", "--workers", "2"
    ])
    assert args.input == r"D:\Manga\Chapter 01"
    assert args.output == r"D:\Out\Chapter 01"
    assert args.workers == 2
