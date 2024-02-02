#include <iostream>

#include "mask_algebra.cuh"
#include <bit>

using namespace std;

int main() {
	using u32 = unsigned int;
	u32 FULL = (1 << 16) - 1;
	for (unsigned int mask = 2; mask <= FULL; mask++) {
		for (int i = 0; i < 16; i++) {
			if (std::popcount(mask) <= i) {
				continue;
			}
			u32 j = e::find_nth_set_bit_v0(mask, i);
			u32 val = e::find_nth_set_bit_v3(mask, i);
			cout << "Checking " << mask << " and i " << i << endl;
			if (val != j) {
				cout << "Expected " << j << " but got " << val << " for mask " << mask << " and query " << i << endl;
				return 1;
			}
		}
	}
	cout << "Checks out" << endl;
	return 0;
}