#pragma unroll
      for(int j=0;j<NT;++j){
        const int idx=nb*NB+j*8+(lane>>2),pos=off+idx;
        const int rowid=pass?pos:(idx<ne?(pair_of[pos]>>3):0);
        const uint4* base=reinterpret_cast<const uint4*>(pass?inter:x);
        const int vi=rowid*(HD/8)+warp*64+q*2;
        if(idx<ne){
          if(pass){
#pragma unroll
            for(int g=0;g<8;++g){uint4 v0=__ldcg(base+vi+g*8),v1=__ldcg(base+vi+g*8+1);
              Bf[j][g][0]=v0.x;Bf[j][g][1]=v0.y;Bf[j][g][2]=v0.z;Bf[j][g][3]=v0.w;
              Bf[j][g][4]=v1.x;Bf[j][g][5]=v1.y;Bf[j][g][6]=v1.z;Bf[j][g][7]=v1.w;}
          }else{
#pragma unroll
            for(int g=0;g<8;++g){uint4 v0=__ldg(base+vi+g*8),v1=__ldg(base+vi+g*8+1);
              Bf[j][g][0]=v0.x;Bf[j][g][1]=v0.y;Bf[j][g][2]=v0.z;Bf[j][g][3]=v0.w;
              Bf[j][g][4]=v1.x;Bf[j][g][5]=v1.y;Bf[j][g][6]=v1.z;Bf[j][g][7]=v1.w;}
          }
        }else{
#pragma unroll
          for(int g=0;g<8;++g){
#pragma unroll
            for(int k=0;k<8;++k)Bf[j][g][k]=0;}
        }
      }
