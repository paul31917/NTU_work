#include <iostream>
#include <stdio.h>
#include <string.h>
using namespace std;

int reg(int num_of_state, int power_of_two){
	if(num_of_state <= power_of_two)
		return 1;
	return 1 + reg(num_of_state, power_of_two*2);
}

int main(){
	char type[10];
	printf("Please input Mealy or Moore.\n");
	scanf("%s", &type);
	while(strcmp(type, "Mealy") != 0 && strcmp(type, "Moore") != 0){
		printf("Please input Mealy or Moore.\n");
		scanf("%s", &type);
	}
	if(strcmp(type, "Mealy") == 0){
		int num_of_state;
		scanf("%d", &num_of_state);

		int num_of_flipflop = reg(num_of_state, 2);
		char type_of_reg[num_of_flipflop];
		int next_of_state_0[num_of_state];
		int next_of_state_1[num_of_state];
		int next_of_out_0[num_of_state];
		int next_of_out_1[num_of_state];
		for(int i = 0; i < num_of_state; i++){
			scanf("%d %d %d", &next_of_state_0[i], &next_of_state_1[i], &next_of_out_0[i], &next_of_out_1[i]);
		}
		printf("What flip-flop do you want to use for %d reg ?\n", num_of_flipflop);
		for(int i = 0; i < num_of_flipflop; i++)
			scanf("%c", &type_of_reg[i]);

		for(int i = 0; i < num_of_flipflop; i++){
			switch(type_of_reg[i]){
				case 'S':
					break;
				case 'J':
					break;
				case 'T':
					break;
				case 'D':
					break;
				default:
					break;
			}
		}
	}
	else{
	}
	return 0;
}
