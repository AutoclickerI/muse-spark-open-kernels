      float acc[NT][4];
      #pragma unroll
      for(int j=0;j<NT;++j)acc[j][0]=acc[j][1]=acc[j][2]=acc[j][3]=0.f;
      #pragma unroll
      for(int h=0;h<WS_GROUPS;++h){
        mbar_wait(dfull+st,ph);
        #pragma unroll
        for(int z=0;z<WS_SUB;++z){
        const int g=h*WS_SUB+z;
        const uint4* ap=reinterpret_cast<const uint4*>(decoded+st*WS_WORDS+z*4096);
        const int idx=nb*NB+(lane>>2);
        const __nv_bfloat16* row=nullptr;
        if(idx<ne)row=pass?inter+(size_t)(off+idx)*HD:x+(size_t)(pair_of[off+idx]>>3)*HD;
        #pragma unroll
        for(int s=0;s<4;++s){
          uint4 av=ap[(s*8+warp)*32+lane];uint2 bv=make_uint2(0,0);
          if(row){const uint2* bp=reinterpret_cast<const uint2*>(row+warp*512+g*64+q*16+s*4);bv=pass?__ldcg(bp):__ldg(bp);}
          mma16816(acc[0],av.x,av.y,av.z,av.w,bv.x,bv.y);
        }
        }
        __syncwarp();if(lane==0)mbar_arrive(dempty+st);
        if(++st==nstage){st=0;ph^=1;}
      }
