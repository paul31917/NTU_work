LOAD DATA LOCAL INFILE '/Users/Apple/desktop/大三上/資料庫管理/hw2/ticket_list'
REPLACE INTO TABLE mydb.Check
FIELDS TERMINATED BY '\t'
lines TERMINATED BY '\n'