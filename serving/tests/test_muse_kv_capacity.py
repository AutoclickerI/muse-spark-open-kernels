"""Bounded SWA must fund every private DFlash row after post-capture resizing."""

from sglang.srt.model_executor.muse_kv_capacity import MuseKVPoolConfigurator


def test_tp8_dflash_budget_and_post_capture_cap():
    plan = MuseKVPoolConfigurator.__new__(MuseKVPoolConfigurator)
    plan._enable_unified_memory = False
    plan._full_per_token = plan._swa_per_token = 256
    plan._full_layers_num, plan._swa_layers_num = 16, 46
    plan._draft_full_layers_num = plan._draft_swa_layers_num = 0
    plan._draft_swa_full_layers_num = 0
    plan._fused_full_entry = None
    plan._draft_cell_size = 1280
    plan._swa_cap = 262657
    for page in (1, 16, 64):
        for budget in (10_000_000_000, 25_000_000_000, 40_000_000_000):
            pools = plan.calculate_pool_sizes(budget, page)
            full, swa = pools.full_max_total_num_tokens, pools.swa_max_total_num_tokens
            assert full % page == swa % page == 0
            assert full * (4096 + 1280) + swa * 11776 <= budget
            assert (full + page) * (4096 + 1280) + swa * 11776 > budget
            assert swa >= plan._swa_cap
            resized = plan.calculate_pool_sizes_from_max_tokens(full - page, page)
            assert resized.full_max_total_num_tokens == full - page
            assert resized.swa_max_total_num_tokens == swa


if __name__ == '__main__':
    test_tp8_dflash_budget_and_post_capture_cap()
    print('TP8 DFlash byte budget and post-capture cap: pass')
