LOAD DATA LOCAL INFILE '/Users/Apple/desktop/大三上/資料庫管理/hw2/user_list.txt'
REPLACE INTO TABLE mydb.Student
FIELDS TERMINATED BY ','
lines TERMINATED BY '\n'
 