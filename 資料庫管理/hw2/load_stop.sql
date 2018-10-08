LOAD DATA LOCAL INFILE '/Users/Apple/desktop/大三上/資料庫管理/hw2/stop_list'
REPLACE INTO TABLE mydb.Stop
FIELDS TERMINATED BY ';'
lines TERMINATED BY '\n'