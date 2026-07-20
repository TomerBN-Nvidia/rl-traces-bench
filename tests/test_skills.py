import os, re, glob
SKILLS = ["run-longtail-bench", "setup-editable-vllm", "interpret-longtail-report", "author-distribution"]
def _fm(path):
    t = open(path).read()
    assert t.startswith("---"), f"{path} missing frontmatter"
    fm = t.split("---", 2)[1]
    assert re.search(r"^name:\s*\S+", fm, re.M) and re.search(r"^description:\s*\S+", fm, re.M)
def test_skills_exist_and_mirror():
    for s in SKILLS:
        c = f".claude/skills/{s}/SKILL.md"; x = f".codex/skills/{s}/SKILL.md"
        assert os.path.exists(c) and os.path.exists(x), s
        _fm(c); _fm(x)
        assert open(c).read() == open(x).read(), f"{s}: claude/codex must be identical"
def test_no_internal_terms_in_skills():
    for p in glob.glob(".c*/skills/**/SKILL.md", recursive=True):
        assert not re.search(r"hsg|nemorl|lustre|coreai|/home/", open(p).read(), re.I)
