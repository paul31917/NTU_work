-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL,ALLOW_INVALID_DATES';


CREATE SCHEMA IF NOT EXISTS `mydb` DEFAULT CHARACTER SET utf8;
USE `mydb` ;


CREATE TABLE IF NOT EXISTS `mydb`.`Student` (
    `Student_id` BIGINT(30) NOT NULL,
    `Name` VARCHAR(255) NOT NULL,
    `Sexual` VARCHAR(255) NOT NULL,
    `major_in` VARCHAR(255) NOT NULL,
    PRIMARY KEY (`Student_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET utf8;

-- -----------------------------------------------------
-- Table `mydb`.`Time_Session`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`Time_Session` (
  `Session` VARCHAR(30) NOT NULL,
  `Price` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`Session`))
ENGINE = InnoDB
DEFAULT CHARACTER SET utf8;


-- -----------------------------------------------------
-- Table `mydb`.`Stop`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`Stop` (
  `Name` VARCHAR(50) NOT NULL,
  `Address` VARCHAR(100) NOT NULL,
  `tot` INT NOT NULL,
  `sbi` INT NOT NULL,
  `bemp` INT NOT NULL,
  PRIMARY KEY (`Name`))
ENGINE = InnoDB
DEFAULT CHARACTER SET utf8;



-- -----------------------------------------------------
-- Table `mydb`.`Check`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`Check` (
  `Check_id` INT NOT NULL,
  `Student_id` BIGINT(30) NOT NULL,
  `Session` VARCHAR(45) NOT NULL,
  `Stop` VARCHAR(45) NOT NULL,
  `Rent_date` DATETIME NOT NULL,
  `Bike_amount` INT NOT NULL,
  PRIMARY KEY (`Check_id`),
  INDEX `fk_Check_Time_Session1_idx` (`Session` ASC),
  INDEX `fk_Check_Student1_idx` (`Student_id` ASC),
  INDEX `fk_Check_Stop1_idx` (`Stop` ASC),
  CONSTRAINT `fk_Check_Time_Session1`
    FOREIGN KEY (`Session`)
    REFERENCES `mydb`.`Time_Session` (`Session`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Check_Student1`
    FOREIGN KEY (`Student_id`)
    REFERENCES `mydb`.`Student` (`Student_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Check_Stop1`
    FOREIGN KEY (`Stop`)
    REFERENCES `mydb`.`Stop` (`Name`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET utf8;



-- -----------------------------------------------------
-- Table `mydb`.`Bike`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`Bike` (
  `Bike_id` INT NOT NULL,
  `廠牌` VARCHAR(45) NOT NULL,
  `規格` VARCHAR(45) NOT NULL,
  `車況` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`Bike_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET utf8;



-- -----------------------------------------------------
-- Table `mydb`.`Record`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `mydb`.`Record` (
  `Record_id` INT NOT NULL,
  `Check_id` INT NOT NULL,
  `Bike_id` INT NOT NULL,
  `取車實際時間` DATETIME NOT NULL,
  `Return_Stop` VARCHAR(45) NULL COMMENT '\n',
  `Return_Time` DATETIME NULL,
  PRIMARY KEY (`Record_id`),
  INDEX `fk_Record_Check1_idx` (`Check_id` ASC),
  INDEX `fk_Record_Stop1_idx` (`Return_Stop` ASC),
  INDEX `fk_Record_Bike1_idx` (`Bike_id` ASC),
  CONSTRAINT `fk_Record_Check1`
    FOREIGN KEY (`Check_id`)
    REFERENCES `mydb`.`Check` (`Check_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Record_Stop1`
    FOREIGN KEY (`Return_Stop`)
    REFERENCES `mydb`.`Stop` (`Name`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_Record_Bike1`
    FOREIGN KEY (`Bike_id`)
    REFERENCES `mydb`.`Bike` (`Bike_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET utf8;



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=0;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
