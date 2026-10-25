import runpy, sys, torch
ns = runpy.run_path(sys.argv[1])
dev = 'cuda'
wo = (torch.randn(8192, 1024, device=dev) * 0.03).to(torch.bfloat16)
lib = ns['ap_lib']('gf')
for window in (2048, 0):
    attn = ns['APAttention'](1, window, dev)
    for trig in (False, True):
        attn.setup(trig)
        st = torch.zeros((512, 16), dtype=torch.int64, device=dev)
        y = torch.empty((1, 8192), dtype=torch.bfloat16, device=dev)
        anchor = torch.zeros(2, dtype=torch.int64, device=dev)
        def run():
            x = attn(trig)
            s = torch.cuda.current_stream().cuda_stream
            e = lib.gf_run(wo.data_ptr(), x.data_ptr(), y.data_ptr(), st.data_ptr(), 1, 8192, 1024, 6, 0, 1, 1, s)
            assert e == 0, e
        for _ in range(3): run()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): run()
        for _ in range(3):
            g.replay(); torch.cuda.synchronize()
        t = st[:, 1::2][:, :7].double()  # globaltimer per mark
        t0 = t[:, 0].min()
        entry = (t[:, 0] - t0) / 1e3; waited = (t[:, 3] - t0) / 1e3; done = (t[:, 6] - t0) / 1e3
        print(f'w={window} trig={trig} entry med/max={entry.median():.2f}/{entry.max():.2f}us '
              f'wait_ret min/med={waited.min():.2f}/{waited.median():.2f}us done max={done.max():.2f}us '
              f'pre-wait loads done med={((t[:,2]-t0)/1e3).median():.2f}')
