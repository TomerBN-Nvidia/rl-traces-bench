import sys

def _load(name):
    from importlib import import_module
    return import_module(f"rl_traces_bench.{name}").main

COMMANDS = {
    "gen-trace": "gen_trace", "analyze": "analyze", "compare": "compare",
    "tau2": "tau2_estimator", "run": "run", "serve": "serve", "doctor": "doctor",
}

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in COMMANDS:
        sys.stderr.write("usage: rl-traces {%s} ...\n" % "|".join(COMMANDS))
        return 2
    return _load(COMMANDS[argv[0]])(argv[1:]) or 0

if __name__ == "__main__":
    raise SystemExit(main())
