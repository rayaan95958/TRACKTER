-- CreateTable
CREATE TABLE "Product" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "mainPartNo" TEXT NOT NULL,
    "description" TEXT NOT NULL,
    "areaInSft" REAL NOT NULL,
    "imageLink" TEXT
);

-- CreateTable
CREATE TABLE "DailyRecord" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "totalSurfaceAreaProduced" INTEGER NOT NULL DEFAULT 0,
    "day" INTEGER NOT NULL,
    "month" INTEGER NOT NULL,
    "year" INTEGER NOT NULL,
    "hour" INTEGER NOT NULL
);

-- CreateTable
CREATE TABLE "MonthlyRecord" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "totalSurfaceAreaProduced" INTEGER NOT NULL DEFAULT 0,
    "day" INTEGER NOT NULL,
    "month" INTEGER NOT NULL,
    "year" INTEGER NOT NULL
);

-- CreateTable
CREATE TABLE "YearlyRecord" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "totalSurfaceAreaProduced" INTEGER NOT NULL DEFAULT 0,
    "year" INTEGER NOT NULL,
    "month" INTEGER NOT NULL
);

-- CreateTable
CREATE TABLE "DailyProductRecord" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "dailyRecordId" INTEGER NOT NULL,
    "productId" INTEGER NOT NULL,
    "partNumber" TEXT NOT NULL,
    "count" INTEGER NOT NULL,
    CONSTRAINT "DailyProductRecord_dailyRecordId_fkey" FOREIGN KEY ("dailyRecordId") REFERENCES "DailyRecord" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "MonthlyProductRecord" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "monthlyRecordId" INTEGER NOT NULL,
    "productId" INTEGER NOT NULL,
    "count" INTEGER NOT NULL,
    "partNumber" TEXT NOT NULL,
    CONSTRAINT "MonthlyProductRecord_monthlyRecordId_fkey" FOREIGN KEY ("monthlyRecordId") REFERENCES "MonthlyRecord" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);

-- CreateTable
CREATE TABLE "YearlyProductRecord" (
    "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    "yearlyRecordId" INTEGER NOT NULL,
    "productId" INTEGER NOT NULL,
    "count" INTEGER NOT NULL,
    "partNumber" TEXT NOT NULL,
    CONSTRAINT "YearlyProductRecord_yearlyRecordId_fkey" FOREIGN KEY ("yearlyRecordId") REFERENCES "YearlyRecord" ("id") ON DELETE RESTRICT ON UPDATE CASCADE
);
