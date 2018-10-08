LOAD DATA LOCAL INFILE '/Users/Apple/desktop/大三上/資料庫管理/hw2/bike.txt'
REPLACE INTO TABLE mydb.Bike
FIELDS TERMINATED BY ';'
lines TERMINATED BY '\n'