if __package__:
    from .cli import main
else:
    # Allow IDEs to run this file directly as well as ``python -m lob_transformer``.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lob_transformer.cli import main


if __name__ == "__main__":
    main()
