@triton.jit
def _ew_immutable_stage(A,B,C,D):
    j=tl.arange(0,128)
    return tl.inline_asm_elementwise('''{
      .shared .align 16 .b8 ew_imm[8192];
      .reg .u32 tid,base,d0,d1,d2,d3;
      mov.u32 tid,%tid.x; mov.u32 base,ew_imm;
      mad.lo.u32 d0,tid,16,base; add.u32 d1,d0,4;add.u32 d2,d0,8;add.u32 d3,d0,12;
      cp.async.ca.shared.global [d0],[$4],4;
      cp.async.ca.shared.global [d1],[$5],4;
      cp.async.ca.shared.global [d2],[$6],4;
      cp.async.ca.shared.global [d3],[$7],4;
      cp.async.commit_group;
      griddepcontrol.wait;
      cp.async.wait_group 0;
      ld.shared.f32 $0,[d0];ld.shared.f32 $1,[d1];ld.shared.f32 $2,[d2];ld.shared.f32 $3,[d3];
    }''',constraints='=f,=f,=f,=f,l,l,l,l',args=[A+j,B+j,C+j,D+j],dtype=(tl.float32,tl.float32,tl.float32,tl.float32),is_pure=False,pack=1)
