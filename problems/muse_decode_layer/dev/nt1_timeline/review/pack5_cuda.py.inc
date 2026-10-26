__device__ __forceinline__ const uint8_t* packet_tile(const uint8_t* bank,size_t idx,int& bytes){
 const uint64_t* offsets=reinterpret_cast<const uint64_t*>(bank);
 const uint64_t a=__ldg(offsets+idx),b=__ldg(offsets+idx+1);bytes=(int)(b-a);return bank+a;
}
__device__ __forceinline__ uint32_t packet_scale(const uint8_t* tile,int index){
 const int packet=index>>8,j=index&255;
 const uint8_t* h=tile+32768;
 const uint32_t raw=*reinterpret_cast<const uint32_t*>(h+16);
 const int start=packet*160+__popc(raw&((1u<<packet)-1u))*96;
 const uint8_t* payload=h+32+start;
 if(raw&(1u<<packet))return payload[j];
 const int bit=j*5,byte=bit>>3,shift=bit&7;
 uint32_t v=payload[byte];if(shift>3)v|=(uint32_t)payload[byte+1]<<8;
 return h[packet]+((v>>shift)&31u);
}
