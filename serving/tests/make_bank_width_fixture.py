"""Small attention/drafter, actual 32x4096 expert banks per serving rank.

Separate agreement fixture: six NVFP4 layers, E32 at TP1 / E64 at TP2.
The standard margin fixture remains the TTFT gate. No harness file is edited.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'serving/harness'))
import make_tiny as m
import torch
from safetensors.torch import save_file


def save(directory,config,weights):
    directory.mkdir(exist_ok=False)
    (directory/'config.json').write_text(json.dumps(config,indent=2))
    save_file({k:v.contiguous().clone() for k,v in weights.items()},str(directory/'model.safetensors'),metadata={'format':'pt'})
    (directory/'harness_manifest.json').write_text(json.dumps({
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'harness_generator_sha256':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),
        'shapes':{k:list(v.shape) for k,v in weights.items()}},indent=2))


def main():
    torch.set_num_threads(4)
    m.HIDDEN=512;m.N_HEADS=8;m.N_LAYERS=6
    m.LAYER_TYPES=['sliding_attention','sliding_attention','full_attention']*2
    m.BF16_EXPERT_LAYERS=()
    m.DRAFT_HEADS=4;m.DRAFT_INTER=1792;m.DRAFT_TARGET_LAYERS=[0,1,2,3,4]
    root=m.common.MODELS
    base_target_weights=m.target_weights;base_draft_weights=m.draft_weights
    for tp,experts in ((1,32),(2,64)):
        m.N_EXPERTS=experts;m.MOE_HIDDEN=m.MOE_INTER=512
        weights=base_target_weights(4421)
        rng=torch.Generator().manual_seed(4921+tp)
        for layer in range(m.N_LAYERS):
            prefix=f'model.language_model.layers.{layer}.mlp'
            weights[prefix+'.pre_expert_proj.weight']=(torch.randn((4096,512),generator=rng)/512**.5).bfloat16()
            weights[prefix+'.post_expert_proj.weight']=(torch.randn((512,4096),generator=rng)/4096**.5).bfloat16()
            for key in ('pre_expert_norm.weight','experts.post_expert_norm.weight'):
                weights[prefix+'.'+key]=(.05*torch.randn(4096,generator=rng)).bfloat16()
            for name,rows in (('gate_up_proj',8192),('down_proj',4096)):
                key=prefix+'.experts.'+name
                weights[key]=torch.randint(256,(experts,rows,2048),dtype=torch.uint8,generator=rng)
                weights[key+'_weight_scale']=torch.ones((experts,rows,256),dtype=torch.float8_e4m3fn)
                weights[key+'_weight_scale_2']=(torch.tensor([1/256,1.7/256]).expand(experts,2).clone()
                    if name=='gate_up_proj' else torch.full((experts,),1/256))
        m.MOE_HIDDEN=m.MOE_INTER=4096
        cfg=m.target_config()
        for kind in ('random','margin'):
            directory=root/f'muse-bank-width-tp{tp}-{kind}'
            if kind=='margin':
                m.target_weights=lambda seed:weights
                weights=m.margin_target_weights(4421)
                m.target_weights=base_target_weights
            save(directory,cfg,weights)
            (directory/'hf_quant_config.json').write_text(json.dumps(m.hf_quant_config(),indent=2))
            (directory/'generation_config.json').write_text(json.dumps({'bos_token_id':m.common.BOS_ID,'eos_token_id':[m.common.EOS_ID,m.common.EOT_ID],'pad_token_id':m.common.PAD_ID}))
            m.write_tokenizer(directory)
            print(directory,sum(v.nbytes for v in weights.values()),flush=True)
        del weights
    draft=base_draft_weights(4422)
    for kind in ('random','margin'):
        if kind=='margin':
            m.draft_weights=lambda seed:draft
            draft=m.margin_draft_weights(4422,4421)
            m.draft_weights=base_draft_weights
        directory=root/f'dflash-bank-width-{kind}'
        save(directory,m.draft_config(),draft)
        print(directory,sum(v.nbytes for v in draft.values()),flush=True)


if __name__=='__main__':main()
