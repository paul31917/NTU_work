CC = gcc
CFLAGS = -Wall

all: receiver sender

receiver: receiver.c
	$(CC) $(CFLAGS) receiver.c -o receiver

sender: sender.c
	$(CC) $(CFLAGS) sender.c -o sender

clean:
	rm -rf receiver sender receiver_log sender_log
