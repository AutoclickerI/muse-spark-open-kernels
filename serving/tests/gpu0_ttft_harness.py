"""Run the unchanged workload on GPU0 with matched <=0.3 memory fractions."""

import integration_harness as integration

base_launch = integration.original_launch


def launch(args, tree, output, port):
    args.mem_fraction_static = 0.3
    return base_launch(args, tree, output, port)


integration.original_launch = launch

# Also supports the existing request-keyed diagnostic events. The final
# admission bracket omits --trace-ttft, so instrumentation never times a gate.
import ttft_harness  # noqa: E402,F401
import run  # noqa: E402


if __name__ == '__main__':
    run.main()
