from poros.cli import views


def _flat(out) -> str:
    return "\n".join("".join(str(part) for part in line) if isinstance(line, list) else str(line)
                     for line in out)


def test_a_flag_description_may_run_onto_the_next_line():
    doc = "Title.\n\nFlags:\n    --store WHERE  stream them\n    from a file\n    --next N  more\n"
    out = _flat(views.verb_help("train", doc, 100))
    assert "from a file" in out and "--next" in out
