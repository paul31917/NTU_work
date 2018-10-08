pragma solidity ^0.4.0;
contract Bet {
    uint startTime;
    uint endTime;
    address oracle = 0xabc;
    address[] player = new address[](4096);
    uint[] playersBet = new uint[](4096);
    uint playerNum = 0;
 
    function Bet() {
        betStart();
    }
    function betStart() {
        startTime = now;   
        endTime = now + 10 minutes;
    }
    function betClose() {
        if(now < endTime || msg.sender != oracle)
            throw;
        winnerWinner();
    }
    function joinBet(uint input)payable{
        player[playerNum] = msg.sender;
        playersBet[playerNum] = input;
        playerNum++;
    }
    function winnerWinner() {
    	uint winner=0;
        for(uint x = 1; x < playerNum; x++) 
        {
        	if(playersBet[x]>playersBet[winner])
        	{
        		winner=x;
        	}
        }
        player[winner].send(this.balance);
		playerNum = 0;
        betStart();
    }
}