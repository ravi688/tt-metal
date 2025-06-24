#include "debug/dprint.h"
#include <concepts>

template<typename T, std::unsigned_integral Addr>
void dprint_array(Addr addr, uint32_t start, uint32_t end)
{
	T* ptr = reinterpret_cast<T*>(addr);
	uint32_t len = end - start;
	if(len > 16)
		end = 7;
	for(uint32_t i = start; i < end; ++i)
		DPRINT << ptr[i] << " ";
	if(len > 16)
	{
		start = end;
		end = len;
		DPRINT << "... ";
	}
	for(uint32_t i = start; i < end; ++i)
		DPRINT << ptr[i] << " ";
	DPRINT << ENDL();
}
