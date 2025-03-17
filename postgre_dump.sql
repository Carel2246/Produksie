-- Drop all tables first to ensure a clean slate (optional, comment out if you want to keep existing data)
DROP TABLE IF EXISTS schedule, template_material, template_task, template, material, calendar, resource_group_association, resource_group, resource, task, job CASCADE;

-- Create independent tables first (no FKs)
CREATE TABLE resource (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    type VARCHAR(1) NOT NULL
);

CREATE TABLE resource_group (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    type VARCHAR(1) NOT NULL
);

CREATE TABLE job (
    id SERIAL PRIMARY KEY,
    job_number VARCHAR(50) NOT NULL UNIQUE,
    description VARCHAR(255),
    order_date TIMESTAMP NOT NULL,
    promised_date TIMESTAMP NOT NULL,
    quantity INTEGER NOT NULL,
    price_each FLOAT NOT NULL,
    customer VARCHAR(100),
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    blocked BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE template (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    description VARCHAR(255),
    price_each REAL DEFAULT 0
);

CREATE TABLE calendar (
    id SERIAL PRIMARY KEY,
    weekday INTEGER NOT NULL UNIQUE,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL
);

-- Create tables with FKs, in dependency order
CREATE TABLE resource_group_association (
    resource_id INTEGER NOT NULL,
    group_id INTEGER NOT NULL,
    PRIMARY KEY (resource_id, group_id),
    FOREIGN KEY (resource_id) REFERENCES resource (id),
    FOREIGN KEY (group_id) REFERENCES resource_group (id)
);

CREATE TABLE task (
    id SERIAL PRIMARY KEY,
    task_number VARCHAR(50) NOT NULL UNIQUE,
    job_number VARCHAR(50) NOT NULL,
    description VARCHAR(255),
    setup_time FLOAT NOT NULL,
    time_each FLOAT NOT NULL,
    predecessors VARCHAR(255),
    resources VARCHAR(255) NOT NULL,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    FOREIGN KEY (job_number) REFERENCES job (job_number)
);

CREATE TABLE schedule (
    id SERIAL PRIMARY KEY,
    task_number VARCHAR(50) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    resources_used VARCHAR(255) NOT NULL,
    FOREIGN KEY (task_number) REFERENCES task (task_number)
);

CREATE TABLE material (
    id SERIAL PRIMARY KEY,
    job_number VARCHAR(50) NOT NULL,
    description VARCHAR(255) NOT NULL,
    quantity FLOAT NOT NULL,
    unit VARCHAR(50) NOT NULL,
    FOREIGN KEY (job_number) REFERENCES job (job_number)
);

CREATE TABLE template_task (
    id SERIAL PRIMARY KEY,
    template_id INTEGER NOT NULL,
    task_number VARCHAR(50) NOT NULL,
    description VARCHAR(255),
    setup_time INTEGER NOT NULL,
    time_each INTEGER NOT NULL,
    predecessors VARCHAR(255),
    resources VARCHAR(255) NOT NULL,
    FOREIGN KEY (template_id) REFERENCES template (id)
);

CREATE TABLE template_material (
    id SERIAL PRIMARY KEY,
    template_id INTEGER NOT NULL,
    description VARCHAR(255) NOT NULL,
    quantity FLOAT NOT NULL,
    unit VARCHAR(50) NOT NULL,
    FOREIGN KEY (template_id) REFERENCES template (id)
);

-- Insert data, following the same order
INSERT INTO resource VALUES
(1,'Andrew','H'),
(2,'Gert','H'),
(3,'James','H'),
(4,'Dean','H'),
(5,'Fernando','H'),
(6,'Ruan','H'),
(7,'Pieter','H'),
(8,'Quinton','H'),
(9,'Thys','H'),
(10,'Vince','H'),
(11,'Wikus','H'),
(12,'Boor1','M'),
(13,'Boor2','M'),
(14,'Cincinatti','M'),
(15,'Cupgun','M'),
(16,'Groot_guillotine','M'),
(17,'Hamburger','M'),
(18,'Klein_buig','M'),
(19,'Klein_guillotine','M'),
(20,'Magboor','M'),
(21,'Promecam','M'),
(22,'Saag1','M'),
(23,'Saag2','M'),
(25,'Verflyn','M');

INSERT INTO resource_group VALUES
(1,'Sweis','H'),
(2,'Handlanger','H'),
(3,'Verf','H'),
(4,'Saag','M'),
(5,'HSaag','H'),
(6,'Grind','H'),
(7,'Boor','M'),
(8,'HBoor','H');

INSERT INTO resource_group_association VALUES
(1,1),(2,1),(3,1),
(1,2),(4,2),(5,2),(3,2),(7,2),(8,2),(6,2),(9,2),(10,2),
(7,3),(8,3),
(22,4),(23,4),
(1,5),(2,5),(3,5),(9,5),(10,5),(11,5),
(4,6),(5,6),(3,6),(7,6),(8,6),(6,6),(9,6),(10,6),
(12,7),(13,7),
(2,8),(3,8),(10,8),(11,8);

INSERT INTO job VALUES
(1,'24330','FIXED RAMP','2025-01-14 22:21:00','2025-03-14 09:30:00',1,141730.0,'FRANS DU TOIT',TRUE,FALSE),
(2,'24356','J800','2025-02-03 08:00:00','2025-03-10 10:00:00',1,19337.0,'PRO PROCESS',FALSE,TRUE),
(3,'24366','LARGE GATE','2025-02-14 08:00:00','2025-03-12 10:00:00',6,3691.0,'PFG',TRUE,FALSE),
(4,'24367','Small Gate','2025-02-14 08:00:00','2025-03-12 10:00:00',4,3089.0,'PFG',FALSE,TRUE),
(5,'24369','Anderson A-raam','2025-02-14 08:00:00','2025-03-12 10:00:00',4,9624.35,'PFG',FALSE,FALSE),
(6,'24368','Anderson Fronts','2025-02-14 08:00:00','2025-03-12 10:00:00',8,1873.0,'PFG',TRUE,FALSE),
(7,'24370','IDC A-raam','2025-02-14 08:00:00','2025-03-12 10:00:00',4,9624.35,'PFG',FALSE,FALSE),
(8,'24371','IDC Fronts','2025-02-14 08:00:00','2025-03-12 10:00:00',8,1873.0,'PFG',TRUE,FALSE),
(9,'24385','Cantilver Raam','2025-02-20 08:00:00','2025-03-06 10:00:00',3,3780.0,'Rackstar',TRUE,FALSE),
(10,'24386','Carbide Bins','2025-02-21 08:00:00','2025-03-10 09:00:00',10,3473.0,'Element 6',TRUE,FALSE),
(11,'24388','Safety Cage','2025-02-27 08:00:00','2025-03-13 10:00:00',1,19430.0,'ARROW',TRUE,FALSE),
(12,'24389','MLR10','2025-03-03 08:00:00','2025-03-17 10:00:00',1,204069.0,'EVEREST STEEL',FALSE,FALSE),
(13,'24392','Cullet Bins','2025-03-01 08:00:00','2025-03-20 10:00:00',3,27120.0,'PFG',TRUE,FALSE),
(14,'24393','Slider Flex Bracket (prototipe)','2025-03-01 08:00:00','2025-03-06 09:00:00',1,733.0,'PFG',TRUE,FALSE),
(15,'24397&8','Tafelblad 1670x650','2025-03-05 07:00:00','2025-03-11 19:38:00',2,1415.0,'Rackstar',TRUE,FALSE),
(16,'25001','880 GRAB','2025-03-06 19:41:00','2025-03-12 17:00:00',1,16127.19,'PFG',FALSE,FALSE),
(17,'25002','Skudmasjien oordoen','2025-03-10 14:37:00','2025-04-07 14:37:00',1,0.0,'ALTYDBO',FALSE,FALSE),
(18,'25003','Nuwe Skudmasjien','2025-03-10 15:03:00','2025-04-14 15:03:00',1,0.0,'ALTYDBO',FALSE,TRUE),
(19,'25004','Grab wassers (3mm)','2025-03-11 15:09:00','2025-03-18 15:09:00',30,19.73,'PFG',FALSE,FALSE),
(20,'25005','Grab wassers (4.5mm)','2025-03-11 15:32:00','2025-03-18 15:32:00',30,23.0,'PFG',FALSE,FALSE),
(21,'25006','Grab wassers (6mm)','2025-03-11 15:33:00','2025-03-18 15:33:00',30,27.30,'PFG',FALSE,FALSE),
(22,'25007','Hek Stoppers','2025-03-12 15:54:00','2025-03-20 15:54:00',3,0.0,'NIGEL METAL',FALSE,FALSE),
(23,'25008','Drupplate (2450 seksie)','2025-03-12 16:14:00','2025-03-18 16:15:00',94,0.0,'Estellae',FALSE,FALSE),
(24,'25009','Cantilver Raam (modifikasie)','2025-03-12 16:36:00','2025-03-13 16:37:00',1,3430.0,'Rackstar',FALSE,FALSE),
(25,'25010','Handrail Brackets','2025-03-13 16:38:00','2025-03-21 16:38:00',6,47.0,'Caldas',FALSE,FALSE),
(26,'25011','Rear Handrail','2025-03-13 17:23:00','2025-03-21 17:23:00',2,2373.0,'Caldas',FALSE,FALSE),
(27,'25012','Tarp Protector Bend Refurb','2025-03-14 17:31:00','2025-03-21 17:31:00',5,197.0,'PFG',FALSE,FALSE),
(28,'25013','Tarp Protector Spring Replacement','2025-03-14 17:34:00','2025-03-21 17:34:00',5,78.70,'PFG',FALSE,FALSE),
(29,'25014','A-frame refurb','2025-03-14 17:41:00','2025-03-21 17:41:00',4,973.0,'PFG',FALSE,FALSE),
(30,'25015','Tarp Prot Straight Refurb','2025-03-14 17:56:00','2025-03-21 17:56:00',2,178.0,'PFG',FALSE,FALSE),
(31,'25016','880 Grab Refurbish','2025-03-14 18:18:00','2025-03-21 18:18:00',1,6505.0,'PFG',FALSE,FALSE),
(32,'25017','Step Ramp Huur','2025-03-17 18:19:00','2025-03-18 18:19:00',1,1575.0,'Tamarin Bay Traders',FALSE,FALSE),
(33,'25018','IPE Boog','2025-03-17 18:46:00','2025-03-20 18:46:00',1,0.0,'Estellae',FALSE,FALSE);

INSERT INTO template VALUES
(1,'J450','J450',0.0),
(2,'880 Refurb','880 Grab Refurbish',6505.0);

INSERT INTO calendar VALUES
(1,0,'07:00:00','16:00:00'),
(2,1,'07:00:00','16:00:00'),
(3,2,'07:00:00','16:00:00'),
(4,3,'07:00:00','16:00:00'),
(5,4,'07:00:00','13:00:00'),
(6,5,'00:00:00','00:00:00'),
(7,6,'00:00:00','00:00:00');

INSERT INTO resource_group_association VALUES
(1,1),(2,1),(3,1),
(1,2),(4,2),(5,2),(3,2),(7,2),(8,2),(6,2),(9,2),(10,2),
(7,3),(8,3),
(22,4),(23,4),
(1,5),(2,5),(3,5),(9,5),(10,5),(11,5),
(4,6),(5,6),(3,6),(7,6),(8,6),(6,6),(9,6),(10,6),
(12,7),(13,7),
(2,8),(3,8),(10,8),(11,8);

INSERT INTO task VALUES(1,'24330-500','24330','Verf',1.0,180.0,'nan','Pieter, Quinton',TRUE);
INSERT INTO task VALUES(2,'24356-10','24356','Sny kante',10.0,20.0,'nan','Gert',FALSE);
INSERT INTO task VALUES(3,'24356-20','24356','Sny body',10.0,10.0,'nan','Gert',FALSE);
INSERT INTO task VALUES(4,'24356-50','24356','Buig body',15.0,10.0,'24356-20','Gert, Andrew, Handlanger',FALSE);
INSERT INTO task VALUES(5,'24356-60','24356','Sny en buig channels',5.0,15.0,'nan','Gert',FALSE);
INSERT INTO task VALUES(6,'24356-70','24356','Sny 10mm parte',5.0,10.0,'nan','Gert',FALSE);
INSERT INTO task VALUES(7,'24356-80','24356','Buig en boor spore',10.0,30.0,'24356-70','Wikus',FALSE);
INSERT INTO task VALUES(8,'24356-100','24356','Bou body',10.0,40.0,'24356-10, 24356-50, 24356-60','Gert, Handlanger',FALSE);
INSERT INTO task VALUES(9,'24356-110','24356','Sweis body',1.0,45.0,'24356-100','Sweis',FALSE);
INSERT INTO task VALUES(10,'24356-120','24356','Grind body',1.0,30.0,'24356-110','Grind',FALSE);
INSERT INTO task VALUES(11,'24356-200','24356','Sny pockets',5.0,10.0,'nan','Gert',FALSE);
INSERT INTO task VALUES(12,'24356-210','24356','Buig pockets',10.0,2.0,'24356-200','Gert, Handlanger',FALSE);
INSERT INTO task VALUES(13,'24356-220','24356','Saag bullets',1.0,5.0,'nan','Saag, HSaag',FALSE);
INSERT INTO task VALUES(14,'24356-230','24356','Draai bullets',10.0,20.0,'24356-220','Quinton',FALSE);
INSERT INTO task VALUES(15,'24356-250','24356','Bou pockets',1.0,15.0,'24356-210, 24356-230','Gert',FALSE);
INSERT INTO task VALUES(16,'24356-260','24356','Sweis pockets',1.0,25.0,'24356-250','Sweis',FALSE);
INSERT INTO task VALUES(17,'24356-270','24356','Grind pockets',1.0,15.0,'24356-260','Grind',FALSE);
INSERT INTO task VALUES(18,'24356-300','24356','Bou bin',10.0,45.0,'24356-120, 24356-270','Wikus, Handlanger',FALSE);
INSERT INTO task VALUES(19,'24356-310','24356','Sweis bin',1.0,60.0,'24356-300','Sweis',FALSE);
INSERT INTO task VALUES(20,'24356-400','24356','Grind bin',1.0,30.0,'24356-310','Grind',FALSE);
INSERT INTO task VALUES(21,'24356-500','24356','Verf bin',1.0,120.0,'24356-400','Pieter',FALSE);
INSERT INTO task VALUES(22,'24366-10','24366','Saag tubing',5.0,5.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(23,'24366-20','24366','Saag round bars',1.0,2.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(24,'24366-30','24366','Sny end caps en gussets',5.0,1.0,'nan','Groot_guillotine, Andrew',TRUE);
INSERT INTO task VALUES(25,'24366-100','24366','Boor tubing',5.0,8.0,'24366-10','Boor, HBoor',TRUE);
INSERT INTO task VALUES(26,'24366-200','24366','Bou hekkie',10.0,30.0,'24366-20, 24366-30, 24366-100','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(27,'24366-300','24366','Sweis hekkie',1.0,30.0,'24366-200','Sweis',TRUE);
INSERT INTO task VALUES(28,'24366-400','24366','Grind hekkie',1.0,30.0,'24366-300','Grind',TRUE);
INSERT INTO task VALUES(29,'24366-500','24366','Verf hekkie',1.0,10.0,'24366-400','Verf',TRUE);
INSERT INTO task VALUES(30,'24366-600','24366','Plak rubber',1.0,10.0,'24366-500','Quinton',TRUE);
INSERT INTO task VALUES(31,'24367-10','24367','Saag tubing',5.0,5.0,'nan','Saag, HSaag',FALSE);
INSERT INTO task VALUES(32,'24367-20','24367','Saag round bars',1.0,2.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(33,'24367-30','24367','Sny end caps en gussets',5.0,1.0,'nan','Groot_guillotine, Andrew',TRUE);
INSERT INTO task VALUES(34,'24367-100','24367','Boor tubing',5.0,8.0,'24367-10','Boor, HBoor',FALSE);
INSERT INTO task VALUES(35,'24367-200','24367','Bou hekkie',10.0,30.0,'24367-20, 24367-30, 24367-100','Wikus, Handlanger',FALSE);
INSERT INTO task VALUES(36,'24367-300','24367','Sweis hekkie',1.0,30.0,'24367-200','Sweis',FALSE);
INSERT INTO task VALUES(37,'24367-400','24367','Grind hekkie',1.0,30.0,'24367-300','Grind',FALSE);
INSERT INTO task VALUES(38,'24367-500','24367','Verf hekkie',1.0,10.0,'24367-400','Verf',FALSE);
INSERT INTO task VALUES(39,'24367-600','24367','Plak rubber',1.0,10.0,'24367-500','Quinton',FALSE);
INSERT INTO task VALUES(40,'24368-10','24368','Saag tubing',5.0,2.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(41,'24368-20','24368','Saag pypies',5.0,5.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(42,'24368-30','24368','Sny plaatparte',5.0,5.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(43,'24368-100','24368','Bou front',10.0,5.0,'24368-10, 24368-20, 24368-30','Wikus',TRUE);
INSERT INTO task VALUES(44,'24368-200','24368','Sweis front',1.0,20.0,'24368-100','Sweis',TRUE);
INSERT INTO task VALUES(45,'24368-300','24368','Grind front',1.0,10.0,'24368-200','Grind',TRUE);
INSERT INTO task VALUES(46,'24368-400','24368','Verf front',1.0,10.0,'24368-300','Verf',TRUE);
INSERT INTO task VALUES(47,'24368-500','24368','Plak rubber',1.0,3.0,'24368-400','Quinton',TRUE);
INSERT INTO task VALUES(48,'24371-10','24371','Saag tubing',5.0,2.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(49,'24371-20','24371','Saag pypies',5.0,5.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(50,'24371-30','24371','Sny plaatparte',5.0,5.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(51,'24371-100','24371','Bou front',10.0,5.0,'24371-10, 24371-20, 24371-30','Wikus',TRUE);
INSERT INTO task VALUES(52,'24371-200','24371','Sweis front',1.0,20.0,'24371-100','Sweis',TRUE);
INSERT INTO task VALUES(53,'24371-300','24371','Grind front',1.0,10.0,'24371-200','Grind',TRUE);
INSERT INTO task VALUES(54,'24371-400','24371','Verf front',1.0,10.0,'24371-300','Verf',TRUE);
INSERT INTO task VALUES(55,'24371-500','24371','Plak rubber',1.0,3.0,'24371-400','Quinton',TRUE);
INSERT INTO task VALUES(56,'24369-10','24369','Saag channels',5.0,5.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(57,'24369-20','24369','Saag tubing',5.0,10.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(58,'24369-30','24369','Sny plaatparte',5.0,5.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(59,'24369-40','24369','Saag pyp',1.0,2.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(60,'24369-110','24369','Plasma channels',5.0,8.0,'24369-10','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(61,'24369-120','24369','Buig plaatparte',10.0,2.0,'24369-30','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(62,'24369-200','24369','Bou in jig',10.0,30.0,'24369-20, 24369-40, 24369-110, 24369-120','Wikus, Handlanger',FALSE);
INSERT INTO task VALUES(63,'24369-300','24369','Sweis A-raam',1.0,30.0,'24369-200','Sweis, Andrew',FALSE);
INSERT INTO task VALUES(64,'24369-400','24369','Grind A-raam',1.0,15.0,'24369-300','Grind, Fernando, Dean',FALSE);
INSERT INTO task VALUES(65,'24369-500','24369','Verf A-raam',1.0,20.0,'24369-400','Pieter, Quinton',FALSE);
INSERT INTO task VALUES(66,'24369-600','24369','Plak rubber',1.0,15.0,'24369-500','Quinton',FALSE);
INSERT INTO task VALUES(67,'24370-10','24370','Saag channels',5.0,5.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(68,'24370-20','24370','Saag tubing',5.0,10.0,'nan','Saag, HSaag',FALSE);
INSERT INTO task VALUES(69,'24370-30','24370','Sny plaatparte',5.0,5.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(70,'24370-40','24370','Saag pyp',1.0,2.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(71,'24370-110','24370','Plasma channels',5.0,8.0,'24370-10','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(72,'24370-120','24370','Buig plaatparte',10.0,2.0,'24370-30','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(73,'24370-200','24370','Bou in jig',10.0,30.0,'24370-20, 24370-40, 24370-110, 24370-120','Wikus, Handlanger',FALSE);
INSERT INTO task VALUES(74,'24370-300','24370','Sweis A-raam',1.0,30.0,'24370-200','Sweis, Andrew',FALSE);
INSERT INTO task VALUES(75,'24370-400','24370','Grind A-raam',1.0,15.0,'24370-300','Grind, Fernando, Dean',FALSE);
INSERT INTO task VALUES(76,'24370-500','24370','Verf A-raam',1.0,20.0,'24370-400','Pieter, Quinton',FALSE);
INSERT INTO task VALUES(77,'24370-600','24370','Plak rubber',1.0,15.0,'24370-500','Quinton',FALSE);
INSERT INTO task VALUES(78,'24385-10','24385','Saag tubing',10.0,30.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(79,'24385-20','24385','Sny plaatparte',5.0,15.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(80,'24385-100','24385','Bou raam',45.0,15.0,'24385-10, 24385-20','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(81,'24385-200','24385','Sweis raam',1.0,30.0,'24385-100','Sweis',TRUE);
INSERT INTO task VALUES(82,'24385-300','24385','Grind raam',1.0,15.0,'24385-200','Grind',TRUE);
INSERT INTO task VALUES(83,'24385-400','24385','Verf raam',1.0,30.0,'24385-300','Verf',TRUE);
INSERT INTO task VALUES(84,'24386-10','24386','Sny plaatparte',5.0,15.0,'nan','Groot_guillotine, Gert, Handlanger',TRUE);
INSERT INTO task VALUES(85,'24386-20','24386','Corrugate parte',10.0,10.0,'24386-10','Gert, Andrew, Handlanger',TRUE);
INSERT INTO task VALUES(86,'24386-30','24386','Buig parte',15.0,10.0,'24386-20','Gert, Andrew, Handlanger',TRUE);
INSERT INTO task VALUES(87,'24386-100','24386','Saag bene',5.0,5.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(88,'24386-200','24386','Sny voete plate',5.0,2.0,'nan','Groot_guillotine, Quinton',TRUE);
INSERT INTO task VALUES(89,'24386-210','24386','Press voete',30.0,1.0,'24386-200','Quinton',TRUE);
INSERT INTO task VALUES(90,'24386-300','24386','Bou bins',30.0,20.0,'24386-30, 24386-100, 24386-210','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(91,'24386-400','24386','Sweis bins',1.0,20.0,'24386-300','Sweis, Andrew',TRUE);
INSERT INTO task VALUES(92,'24386-500','24386','Grind bins',1.0,15.0,'24386-400','Grind, Fernando, Dean',TRUE);
INSERT INTO task VALUES(93,'24386-600','24386','Verf bins',1.0,30.0,'24386-500','Pieter',TRUE);
INSERT INTO task VALUES(94,'24388-10','24388','Saag tubing',5.0,45.0,'nan','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(95,'24388-20','24388','Sny plaatparte',5.0,45.0,'nan','Groot_guillotine, Gert, Handlanger',TRUE);
INSERT INTO task VALUES(96,'24388-30','24388','Sny mesh',1.0,20.0,'nan','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(97,'24388-100','24388','Bou cage',30.0,180.0,'24388-10, 24388-20','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(98,'24388-200','24388','Sweis cage',1.0,90.0,'24388-100, 24388-30','Sweis',TRUE);
INSERT INTO task VALUES(99,'24388-300','24388','Grind cage',1.0,60.0,'24388-200','Grind, Fernando, Dean',TRUE);
INSERT INTO task VALUES(100,'24388-400','24388','Verf cage',1.0,90.0,'24388-300','Pieter',TRUE);
INSERT INTO task VALUES(101,'24389-10','24389','Plasma tubings',15.0,570.0,'nan','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(102,'24389-11','24389','Grind plasma tubes',1.0,180.0,'24389-10','Grind, Fernando, Dean',TRUE);
INSERT INTO task VALUES(103,'24389-12','24389','Sny plaatparte',10.0,45.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(104,'24389-20','24389','Saag tubing',30.0,360.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(105,'24389-30','24389','Bou ribbes',30.0,510.0,'24389-11, 24389-20','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(106,'24389-40','24389','Sweis ribbes',1.0,300.0,'24389-30','Sweis, Andrew',TRUE);
INSERT INTO task VALUES(107,'24389-50','24389','Grind ribbes',1.0,180.0,'24389-40','Grind, Fernando, Dean',TRUE);
INSERT INTO task VALUES(108,'24389-100','24389','Bou raam',45.0,360.0,'24389-50','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(109,'24389-200','24389','Sweis raam',1.0,300.0,'24389-100','Sweis, Andrew',TRUE);
INSERT INTO task VALUES(110,'24389-300','24389','Sit grating op',1.0,180.0,'24389-200','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(111,'24389-400','24389','Bou flappe',10.0,120.0,'24389-12','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(112,'24389-410','24389','Bou onderstel',15.0,60.0,'24389-20','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(113,'24389-420','24389','Sweis flappe',1.0,30.0,'24389-400','Sweis',TRUE);
INSERT INTO task VALUES(114,'24389-430','24389','Sweis onderstel',1.0,20.0,'24389-410','Sweis',TRUE);
INSERT INTO task VALUES(115,'24389-500','24389','Sit raam aanmekaar',1.0,60.0,'24389-300, 24389-420, 24389-430','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(116,'24389-600','24389','Verf raam',20.0,510.0,'24389-500','Verf',FALSE);
INSERT INTO task VALUES(117,'24392-10','24392','Sny kante',10.0,20.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(118,'24392-20','24392','Sny body',10.0,10.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(119,'24392-50','24392','Buig body',15.0,10.0,'24392-20','Gert, Andrew, Handlanger',TRUE);
INSERT INTO task VALUES(120,'24392-60','24392','Sny en buig channels',5.0,15.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(121,'24392-70','24392','Sny 10mm parte',5.0,10.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(122,'24392-80','24392','Buig en boor spore',10.0,30.0,'24392-70','Wikus',TRUE);
INSERT INTO task VALUES(123,'24392-100','24392','Bou body',10.0,40.0,'24392-10, 24392-50, 24392-60','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(124,'24392-110','24392','Sweis body',1.0,45.0,'24392-100','Sweis',TRUE);
INSERT INTO task VALUES(125,'24392-120','24392','Grind body',1.0,30.0,'24392-110','Grind',TRUE);
INSERT INTO task VALUES(126,'24392-200','24392','Sny pockets',5.0,10.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(127,'24392-210','24392','Buig pockets',10.0,2.0,'24392-200','Gert, Handlanger',TRUE);
INSERT INTO task VALUES(128,'24392-220','24392','Saag bullets',1.0,5.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(129,'24392-230','24392','Draai bullets',10.0,20.0,'24392-220','Quinton',TRUE);
INSERT INTO task VALUES(130,'24392-250','24392','Bou pockets',1.0,15.0,'24392-210, 24392-230','Gert',TRUE);
INSERT INTO task VALUES(131,'24392-260','24392','Sweis pockets',1.0,25.0,'24392-250','Sweis',TRUE);
INSERT INTO task VALUES(132,'24392-270','24392','Grind pockets',1.0,15.0,'24392-260','Grind',TRUE);
INSERT INTO task VALUES(133,'24392-300','24392','Bou bin',10.0,45.0,'24392-120, 24392-270','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(134,'24392-310','24392','Sweis bin',1.0,60.0,'24392-300','Sweis',TRUE);
INSERT INTO task VALUES(135,'24392-400','24392','Grind bin',1.0,30.0,'24392-310','Grind',TRUE);
INSERT INTO task VALUES(136,'24392-500','24392','Verf bin',1.0,120.0,'24392-400','Pieter',TRUE);
INSERT INTO task VALUES(137,'24393-10','24393','Saag EA',1.0,10.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(138,'24393-20','24393','Saag pyp',1.0,2.0,'nan','Saag, HSaag',TRUE);
INSERT INTO task VALUES(139,'24393-30','24393','Sny plaatparte',5.0,5.0,'nan','Gert',TRUE);
INSERT INTO task VALUES(140,'24393-100','24393','Bou T-nuts',10.0,2.0,'24393-30','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(141,'24393-110','24393','Sweis T-nuts',1.0,3.0,'24393-100','Sweis',TRUE);
INSERT INTO task VALUES(142,'24393-200','24393','Bou bracket',10.0,4.0,'24393-10, 24393-20, 24393-30','Wikus, Handlanger',TRUE);
INSERT INTO task VALUES(143,'24393-210','24393','Sweis bracket',1.0,3.0,'24393-200','Sweis',TRUE);
INSERT INTO task VALUES(144,'24393-300','24393','Grind',1.0,10.0,'24393-110, 24393-210','Grind',TRUE);
INSERT INTO task VALUES(145,'24393-400','24393','Verf',1.0,5.0,'24393-300','Pieter',TRUE);
INSERT INTO task VALUES(146,'24397-10','24397&8','Plasma',5.0,5.0,'','Gert',TRUE);
INSERT INTO task VALUES(147,'24397-20','24397&amp;8','Skoonmaak',1.0,5.0,'24397-10','Grind',TRUE);
INSERT INTO task VALUES(148,'25001-1','25001','Saag raam',10.0,30.0,'','Saag, HSaag',TRUE);
INSERT INTO task VALUES(149,'25001-2','25001','Plasma',1.0,15.0,'','Gert',TRUE);
INSERT INTO task VALUES(150,'25001-3','25001','Bou raam',10.0,90.0,'25001-1, 25001-2','Wikus, Vince',TRUE);
INSERT INTO task VALUES(151,'25001-4','25001','Sweis raam',1.0,60.0,'25001-3','Andrew',TRUE);
INSERT INTO task VALUES(152,'25001-5','25001','Saag los parte',5.0,30.0,'','Saag, HSaag',TRUE);
INSERT INTO task VALUES(153,'25001-6','25001','Sit los parte op',1.0,60.0,'25001-4, 25001-5','Wikus, Vince',TRUE);
INSERT INTO task VALUES(154,'25001-7','25001','Sweis grab volledig',1.0,30.0,'25001-6','Sweis',TRUE);
INSERT INTO task VALUES(155,'25001-8','25001','Grind',1.0,30.0,'25001-7','Dean, Grind',TRUE);
INSERT INTO task VALUES(156,'25001-9','25001','Verf',10.0,60.0,'25001-8','Quinton, Pieter',TRUE);
INSERT INTO task VALUES(157,'25001-10','25001','Plak rubber',1.0,20.0,'25001-9','Quinton',FALSE);
INSERT INTO task VALUES(158,'25001-11','25001','Maak grab klaar',1.0,45.0,'25001-10','Quinton, Pieter',FALSE);
INSERT INTO task VALUES(159,'25002-1','25002','Strip',1.0,60.0,'','James',TRUE);
INSERT INTO task VALUES(160,'25002-2','25002','Skoonmaak',1.0,120.0,'25002-1','Quintin, Handlanger',FALSE);
INSERT INTO task VALUES(161,'25002-3','25002','Deurgaan en regmaak',1.0,180.0,'25002-2','Wikus, Handlanger',FALSE);
INSERT INTO task VALUES(162,'25002-4','25002','Oorverf',1.0,120.0,'25002-3','Verf',FALSE);
INSERT INTO task VALUES(163,'25002-5','25002','Aanmekaarsit',1.0,60.0,'25002-4','Wikus, James',FALSE);
INSERT INTO task VALUES(164,'25003-1','25003','Plasma plaatparte',15.0,30.0,'','Gert',FALSE);
INSERT INTO task VALUES(165,'25003-2','25003','Saag seksies',1.0,60.0,'','HSaag, Saag',FALSE);
INSERT INTO task VALUES(166,'25003-3','25003','Berei parte voor',1.0,60.0,'25003-1, 25003-2','Wikus, Handlanger',FALSE);
INSERT INTO task VALUES(167,'25003-4','25003','Bou skudmasjien',1.0,180.0,'25003-3','Wikus, Handlanger',FALSE);
INSERT INTO task VALUES(168,'25003-5','25003','Sweis',1.0,120.0,'25003-4','Andrew',FALSE);
INSERT INTO task VALUES(169,'25003-6','25003','Verf',1.0,120.0,'25003-5','Verf',FALSE);
INSERT INTO task VALUES(170,'25003-7','25003','Sit aanmekaar',1.0,240.0,'25003-6','Wikus, Gert',FALSE);
INSERT INTO task VALUES(171,'25004-1','25004','Plasma',1.0,1.0,'','Gert',TRUE);
INSERT INTO task VALUES(172,'25004-2','25004','Grind',1.0,1.0,'25004-1','Grinder',TRUE);
INSERT INTO task VALUES(173,'25004-3','25004','Verf',1.0,1.0,'25004-2','Verf',FALSE);
INSERT INTO task VALUES(174,'25005-1','25005','Plasma',1.0,1.0,'','Gert',TRUE);
INSERT INTO task VALUES(175,'25005-2','25005','Grind',1.0,1.0,'25005-1','Grinder',TRUE);
INSERT INTO task VALUES(176,'25005-3','25005','Verf',1.0,1.0,'25005-2','Verf',FALSE);
INSERT INTO task VALUES(180,'25007-1','25007','Sny plaatjies',1.0,5.0,'','Gert',FALSE);
INSERT INTO task VALUES(181,'25007-2','25007','Sit plaatjies op',10.0,10.0,'25007-1','Gert, Handlanger',FALSE);
INSERT INTO task VALUES(182,'25007-3','25007','Verf',10.0,10.0,'25007-2','Pieter, Quintin',FALSE);
INSERT INTO task VALUES(185,'25009-1','25009','Modifikasie',1.0,480.0,'','Wikus, Gert, Andrew, Vince',TRUE);
INSERT INTO task VALUES(186,'25010-1','25010','Plasma',5.0,1.0,'','Gert',FALSE);
INSERT INTO task VALUES(187,'25010-2','25010','Skoonmaak',1.0,2.0,'25010-1','Grinder',FALSE);
INSERT INTO task VALUES(188,'25010-3','25010','Buig',5.0,0.5,'25010-2','Gert',FALSE);
INSERT INTO task VALUES(189,'25010-4','25010','Verf',1.0,2.0,'25010-3','Verf',FALSE);
INSERT INTO task VALUES(190,'25011-1','25011','Saag',5.0,1.0,'','Saag, HSaag',FALSE);
INSERT INTO task VALUES(191,'25011-2','25011','Plasma',5.0,5.0,'','Gert',FALSE);
INSERT INTO task VALUES(192,'25011-3','25011','Buig',10.0,0.5,'25011-2','Gert, Handlanger',FALSE);
INSERT INTO task VALUES(193,'25011-4','25011','Bou Raam',10.0,10.0,'25011-1, 25011-3','Gert, Handlanger',FALSE);
INSERT INTO task VALUES(194,'25011-5','25011','Sweis',1.0,15.0,'25011-4','Sweis',FALSE);
INSERT INTO task VALUES(195,'25011-6','25011','Grind',1.0,10.0,'25011-5','Grind',FALSE);
INSERT INTO task VALUES(196,'25011-7','25011','Verf',5.0,15.0,'25011-6','Pieter, Quinton',FALSE);
INSERT INTO task VALUES(197,'25012-1','25012','Deurgaan',1.0,2.0,'','Wikus',FALSE);
INSERT INTO task VALUES(198,'25012-2','25012','Regmaak',10.0,5.0,'','Gert, Handlanger',FALSE);
INSERT INTO task VALUES(199,'25013-1','25013','Bou komponente',1.0,5.0,'','Gert',FALSE);
INSERT INTO task VALUES(200,'25013-2','25013','Tack op',1.0,5.0,'25013-1','Gert',FALSE);
INSERT INTO task VALUES(201,'25013-3','25013','Verf',1.0,10.0,'25013-2','Verf',FALSE);
INSERT INTO task VALUES(202,'25014-1','25014','Deurgaan',5.0,5.0,'','Wikus',FALSE);
INSERT INTO task VALUES(203,'25014-2','25014','Regmaak',5.0,30.0,'25014-1','James',FALSE);
INSERT INTO task VALUES(204,'25014-3','25014','Verf',1.0,20.0,'25014-2','Verf',FALSE);
INSERT INTO task VALUES(205,'25015-1','25015','Deurgaan',5.0,2.0,'','Wikus',FALSE);
INSERT INTO task VALUES(206,'25015-2','25015','Regmaak',1.0,10.0,'25015-1','James',FALSE);
INSERT INTO task VALUES(207,'25015-3','25015','Verf',5.0,5.0,'25015-2','Verf',FALSE);
INSERT INTO task VALUES(208,'25016-1','25016','Strip',5.0,30.0,'','Handlanger',FALSE);
INSERT INTO task VALUES(209,'25016-2','25016','Deurgaan',1.0,10.0,'25016-1','Wikus',FALSE);
INSERT INTO task VALUES(210,'25016-3','25016','Regmaak',10.0,60.0,'25016-2','Gert, Andrew',FALSE);
INSERT INTO task VALUES(211,'25016-4','25016','Sweis',1.0,45.0,'25016-3','Andrew',FALSE);
INSERT INTO task VALUES(212,'25016-5','25016','Verf',10.0,60.0,'25016-4','Pieter, Quinton',FALSE);
INSERT INTO task VALUES(213,'25017-1','25017','Skoonmaak',1.0,45.0,'','Pieter, Quinton',FALSE);
INSERT INTO task VALUES(214,'25018-1','25018','Saag',5.0,15.0,'','Saag,HSaag',FALSE);
INSERT INTO task VALUES(215,'25018-2','25018','Buig',30.0,240.0,'25018-1','James',FALSE);
INSERT INTO task VALUES(216,'25006-1','25006','Plasma',1.0,1.0,'','',TRUE);
INSERT INTO task VALUES(217,'25006-2','25006','Grind',1.0,1.0,'25006-1','',TRUE);
INSERT INTO task VALUES(218,'25006-3','25006','Verf',1.0,1.0,'25006-2','',FALSE);
INSERT INTO task VALUES(219,'25008-1','25008','Sny plaat',5.0,0.5,'','',TRUE);
INSERT INTO task VALUES(220,'25008-2','25008','Buig',15.0,0.2999999999999999889,'25008-1','',TRUE);

INSERT INTO schedule VALUES(1,'24369-200','2025-03-18 07:00:00.000000','2025-03-18 12:45:00.000000','Wikus, James');
INSERT INTO schedule VALUES(2,'24369-300','2025-03-19 07:00:00.000000','2025-03-19 07:56:00.000000','Gert, Andrew');
INSERT INTO schedule VALUES(3,'24369-400','2025-03-19 07:00:00.000000','2025-03-19 10:58:00.000000','James, Fernando, Dean');
INSERT INTO schedule VALUES(4,'24369-500','2025-03-20 07:00:00.000000','2025-03-20 11:36:00.000000','Pieter, Quinton');
INSERT INTO schedule VALUES(5,'24369-600','2025-03-20 07:00:00.000000','2025-03-20 13:58:00.000000','Quinton');
INSERT INTO schedule VALUES(6,'24370-20','2025-03-18 11:02:00.000000','2025-03-18 12:14:00.000000','Saag2, Thys');
INSERT INTO schedule VALUES(7,'24370-200','2025-03-24 07:00:00.000000','2025-03-24 10:11:00.000000','Wikus, Pieter');
INSERT INTO schedule VALUES(8,'24370-300','2025-03-24 07:00:00.000000','2025-03-24 14:22:00.000000','Gert, Andrew');
INSERT INTO schedule VALUES(9,'24370-400','2025-03-25 07:00:00.000000','2025-03-25 08:24:00.000000','Vince, Fernando, Dean');
INSERT INTO schedule VALUES(10,'24370-500','2025-03-25 07:00:00.000000','2025-03-25 10:46:00.000000','Pieter, Quinton');
INSERT INTO schedule VALUES(11,'24370-600','2025-03-25 07:00:00.000000','2025-03-25 13:08:00.000000','Quinton');
INSERT INTO schedule VALUES(12,'24389-600','2025-03-19 07:00:00.000000','2025-03-19 10:25:00.000000','Pieter');
INSERT INTO schedule VALUES(13,'25001-10','2025-03-18 11:10:00.000000','2025-03-18 12:06:00.000000','Quinton');
INSERT INTO schedule VALUES(14,'25001-11','2025-03-20 07:00:00.000000','2025-03-20 13:43:00.000000','Quinton, Pieter');
INSERT INTO schedule VALUES(15,'25002-2','2025-03-19 07:42:00.000000','2025-03-19 15:50:00.000000','Andrew');
INSERT INTO schedule VALUES(16,'25002-3','2025-03-21 07:00:00.000000','2025-03-21 12:52:00.000000','Wikus, Quinton');
INSERT INTO schedule VALUES(17,'25002-4','2025-03-24 07:00:00.000000','2025-03-24 11:54:00.000000','Quinton');
INSERT INTO schedule VALUES(18,'25002-5','2025-03-24 07:00:00.000000','2025-03-24 14:56:00.000000','Wikus, James');
INSERT INTO schedule VALUES(19,'25004-3','2025-03-18 11:31:00.000000','2025-03-18 12:58:00.000000','Quinton');
INSERT INTO schedule VALUES(20,'25005-3','2025-03-21 07:00:00.000000','2025-03-21 10:22:00.000000','Pieter');
INSERT INTO schedule VALUES(21,'25007-1','2025-03-18 12:10:00.000000','2025-03-18 14:01:00.000000','Gert');
INSERT INTO schedule VALUES(22,'25007-2','2025-03-19 07:00:00.000000','2025-03-19 12:57:00.000000','Gert, Dean');
INSERT INTO schedule VALUES(23,'25007-3','2025-03-24 07:00:00.000000','2025-03-24 13:01:00.000000','Pieter');
INSERT INTO schedule VALUES(24,'25010-1','2025-03-18 11:10:00.000000','2025-03-18 11:56:00.000000','Gert');
INSERT INTO schedule VALUES(25,'25010-2','2025-03-18 11:21:00.000000','2025-03-18 12:20:00.000000','');
INSERT INTO schedule VALUES(26,'25010-3','2025-03-18 11:36:00.000000','2025-03-18 12:45:00.000000','Gert');
INSERT INTO schedule VALUES(27,'25010-4','2025-03-21 07:00:00.000000','2025-03-21 11:06:00.000000','Pieter');
INSERT INTO schedule VALUES(28,'25011-1','2025-03-18 10:55:00.000000','2025-03-18 11:22:00.000000','Saag2, Vince');
INSERT INTO schedule VALUES(29,'25011-2','2025-03-18 11:21:00.000000','2025-03-18 12:22:00.000000','Gert');
INSERT INTO schedule VALUES(30,'25011-3','2025-03-18 12:45:00.000000','2025-03-18 15:06:00.000000','Gert, James');
INSERT INTO schedule VALUES(31,'25011-4','2025-03-19 08:02:00.000000','2025-03-19 14:59:00.000000','Gert, Thys');
INSERT INTO schedule VALUES(32,'25011-5','2025-03-20 07:00:00.000000','2025-03-20 09:22:00.000000','Andrew');
INSERT INTO schedule VALUES(33,'25011-6','2025-03-20 07:00:00.000000','2025-03-20 10:14:00.000000','Vince');
INSERT INTO schedule VALUES(34,'25011-7','2025-03-24 07:00:00.000000','2025-03-24 15:58:00.000000','Pieter, Quinton');
INSERT INTO schedule VALUES(35,'25012-1','2025-03-18 13:05:00.000000','2025-03-18 15:46:00.000000','Wikus');
INSERT INTO schedule VALUES(36,'25012-2','2025-03-18 07:00:00.000000','2025-03-18 11:10:00.000000','Gert, Quinton');
INSERT INTO schedule VALUES(37,'25013-1','2025-03-18 11:44:00.000000','2025-03-18 13:19:00.000000','Gert');
INSERT INTO schedule VALUES(38,'25013-2','2025-03-19 07:36:00.000000','2025-03-19 14:03:00.000000','Gert');
INSERT INTO schedule VALUES(39,'25013-3','2025-03-24 07:00:00.000000','2025-03-24 14:32:00.000000','Pieter');
INSERT INTO schedule VALUES(40,'25014-1','2025-03-19 07:00:00.000000','2025-03-19 07:22:00.000000','Wikus');
INSERT INTO schedule VALUES(41,'25014-2','2025-03-19 07:00:00.000000','2025-03-19 14:04:00.000000','James');
INSERT INTO schedule VALUES(42,'25014-3','2025-03-21 07:00:00.000000','2025-03-21 12:40:00.000000','Pieter');
INSERT INTO schedule VALUES(43,'25015-1','2025-03-18 12:56:00.000000','2025-03-18 15:26:00.000000','Wikus');
INSERT INTO schedule VALUES(44,'25015-2','2025-03-18 13:05:00.000000','2025-03-18 15:56:00.000000','James');
INSERT INTO schedule VALUES(45,'25015-3','2025-03-20 07:00:00.000000','2025-03-20 14:44:00.000000','Pieter');
INSERT INTO schedule VALUES(46,'25016-1','2025-03-18 07:00:00.000000','2025-03-18 11:10:00.000000','Andrew');
INSERT INTO schedule VALUES(47,'25016-2','2025-03-18 12:45:00.000000','2025-03-18 15:06:00.000000','Wikus');
INSERT INTO schedule VALUES(48,'25016-3','2025-03-19 07:00:00.000000','2025-03-19 11:07:00.000000','Gert, Andrew');
INSERT INTO schedule VALUES(49,'25016-4','2025-03-19 07:00:00.000000','2025-03-19 13:03:00.000000','Andrew');
INSERT INTO schedule VALUES(50,'25016-5','2025-03-21 07:00:00.000000','2025-03-21 08:41:00.000000','Pieter, Quinton');
INSERT INTO schedule VALUES(51,'25017-1','2025-03-20 07:00:00.000000','2025-03-20 15:45:00.000000','Pieter, Quinton');
INSERT INTO schedule VALUES(52,'25018-1','2025-03-18 07:00:00.000000','2025-03-18 10:55:00.000000','Saag2, Vince');
INSERT INTO schedule VALUES(53,'25018-2','2025-03-20 07:00:00.000000','2025-03-20 11:39:00.000000','James');
INSERT INTO schedule VALUES(54,'25006-3','2025-03-18 07:00:00.000000','2025-03-18 11:06:00.000000','');

INSERT INTO material VALUES
(1,'25004','3mm Plaat',0.1,'2450x1225'),
(3,'25005','4.5mm Plaat',0.1,'2500x1200'),
(4,'25007','3mm / 4.5mm / 6mm',3.0,'off-cut soos nodig'),
(6,'25010','6mm Plaat',0.1,'2500x1200'),
(7,'25011','4.5mm Plaat',0.25,'2500x1200'),
(8,'25011','42.9x2.5 Rount Tube',4.0,'6m lengte'),
(9,'25013','Veer',5.0,'elk'),
(10,'25013','M10 spring washer',5.0,'elk'),
(11,'25013','10mm Rond',250.0,'mm'),
(12,'25006','6mm Plaat',0.1,'2500x1200'),
(13,'25008','1mm Galv Plaat',14.0,'2450');

INSERT INTO template_task VALUES(1,1,'1','Sny kante',10,10,'','Gert');
INSERT INTO template_task VALUES(2,1,'2','Sny body',10,3,'','Gert, Handlanger');
INSERT INTO template_task VALUES(3,1,'3','Buig body',15,10,'2','Gert, Andrew, Handlanger');
INSERT INTO template_task VALUES(4,1,'4','Sny en buig channels',5,15,'','Wikus');
INSERT INTO template_task VALUES(5,1,'5','Sny 10mm parte',5,10,'','Gert');
INSERT INTO template_task VALUES(6,1,'6','Buig en boor spore',10,30,'5','Wikus');
INSERT INTO template_task VALUES(7,1,'7','Bou body',10,40,'1, 3, 4','Gert, Handlanger');
INSERT INTO template_task VALUES(8,1,'8','Sweis body',5,40,'7','Sweis');
INSERT INTO template_task VALUES(9,1,'9','Grind body',1,30,'8','Grind');
INSERT INTO template_task VALUES(10,1,'10','Sny pockets',5,5,'','Gert');
INSERT INTO template_task VALUES(11,1,'11','Buig pockets',10,2,'10','Gert, Handlanger');
INSERT INTO template_task VALUES(12,1,'12','Saag bullets',1,5,'','Saag, HSaag');
INSERT INTO template_task VALUES(13,1,'13','Draai bullets',10,20,'12','Quinton');
INSERT INTO template_task VALUES(14,1,'14','Bou pockets',1,15,'11, 13','Gert');
INSERT INTO template_task VALUES(15,1,'15','Sweis pockets',1,25,'14','Sweis');
INSERT INTO template_task VALUES(16,1,'16','Grind pockets',1,15,'15','Grind');
INSERT INTO template_task VALUES(17,1,'17','Bou bin',10,45,'6, 9, 16 ','Wikus');
INSERT INTO template_task VALUES(18,1,'18','Sweis bin',1,45,'17','Sweis');
INSERT INTO template_task VALUES(19,1,'19','Grind bin',1,30,'18','Grind');
INSERT INTO template_task VALUES(20,1,'20','Verf bin',1,120,'19','Verf');
INSERT INTO template_task VALUES(21,2,'1','Strip',5,30,'','Handlanger');
INSERT INTO template_task VALUES(22,2,'2','Deurgaan',1,10,'1','Wikus');
INSERT INTO template_task VALUES(23,2,'3','Regmaak',10,60,'2','Gert, Andrew');
INSERT INTO template_task VALUES(24,2,'4','Sweis',1,45,'3','Andrew');
INSERT INTO template_task VALUES(25,2,'5','Verf',10,60,'4','Pieter, Quinton');

INSERT INTO template_material VALUES
(1,1,'3mm Plaat',2.0,'2450x1225'),
(2,1,'4.5mm Plaat',1.0,'2500x1200'),
(3,1,'25mm Bright Bar',300.0,'mm'),
(4,1,'10mm Plaat',0.1,'2500x1200'),
(5,1,'6mm Ketting',3.0,'m'),
(6,1,'8mm carabine hook',2.0,'ea');

-- Set sequence values to match max IDs (prevents ID conflicts on future inserts)
SELECT setval('resource_id_seq', (SELECT MAX(id) FROM resource));
SELECT setval('resource_group_id_seq', (SELECT MAX(id) FROM resource_group));
SELECT setval('job_id_seq', (SELECT MAX(id) FROM job));
SELECT setval('template_id_seq', (SELECT MAX(id) FROM template));
SELECT setval('calendar_id_seq', (SELECT MAX(id) FROM calendar));
SELECT setval('task_id_seq', (SELECT MAX(id) FROM task));
SELECT setval('schedule_id_seq', (SELECT MAX(id) FROM schedule));
SELECT setval('material_id_seq', (SELECT MAX(id) FROM material));
SELECT setval('template_task_id_seq', (SELECT MAX(id) FROM template_task));
SELECT setval('template_material_id_seq', (SELECT MAX(id) FROM template_material));