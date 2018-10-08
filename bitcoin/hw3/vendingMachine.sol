pragma solidity ^0.4.0;
contract vendingMachine {
    address buyer;
    address customer;
    mapping (uint => uint) item_list;
    uint hat = 1;
    uint shirt = 2;
    uint shoes = 3;
    uint hatprice = 300;
    uint shirtprice = 400;
    uint shoesprice = 1000;
    uint buyersChoice=0;
    uint buyersCash=0;
    function vendingMachine() {
        item_list[hat] = hatprice;
        item_list[shirt] = shirtprice;
        item_list[shoes] = shoesprice;
        customer = 0x123;
    }
    function selectItem(uint item) {
        if(buyersCash > 0){
            if(item_list[item] <=buyersCash && buyer == msg.sender){
                uint change = buyersCash - item_list[item];
                customer.send(change);
                giveItemToBuyer();
            }
        }
        else{
            buyersChoice = item;
            buyer = msg.sender;
        }
        //show you the price
    }
    function pay() payable{
        if(buyersChoice == 0){
            buyersCash = msg.value;
            buyer = msg.sender;
        }
        else if(item_list[buyersChoice] <= msg.value && buyer == msg.sender) {
            uint change = msg.value - item_list[buyersChoice];
            customer.send(change);
            giveItemToBuyer();
        }
    }
    function giveItemToBuyer() {
        
    }
}