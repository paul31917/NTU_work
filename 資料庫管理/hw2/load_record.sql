LOAD DATA LOCAL INFILE '/Users/Apple/desktop/大三上/資料庫管理/hw2/record.txt'
REPLACE INTO TABLE mydb.Record
FIELDS TERMINATED BY ';'
lines TERMINATED BY '\n'