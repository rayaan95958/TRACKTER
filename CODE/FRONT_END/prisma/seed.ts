const { PrismaClient } = require('@prisma/client');
const { randomInt } = require('crypto');

const prisma = new PrismaClient();

const products = [
    { mainPartNo: '40515-926-01-FG', description: 'COVER-PUR', areaInSft: 1.71, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-011-51-FG', description: 'ENS CAJA TAP BOX 1200-1600A', areaInSft: 45.37, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-030-13-FG', description: 'BOX BLANK 1200A EUSERC', areaInSft: 29.35, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-094-52-FG', description: 'ASSY BOTTOM EUSERC 1PH HYBRID-PUR', areaInSft: 3.81, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-094-01-FG', description: 'RESPALDO EUSERC 3PH HYBRID -PUR', areaInSft: 24.67, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-094-52-FG', description: 'ASSY BOTTOM EUSERC 1PH HYBRID-PUR', areaInSft: 3.81, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-095-50-FG', description: 'ASSY BOTTOM EUSERC 3PH HYBRID', areaInSft: 4.93, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-098-02-FG', description: 'BOTTOM COVER FUSE 1PH HYBRID-PUR', areaInSft: 9.50, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-116-02-FG', description: 'BOTTOM COVER BKR 3PH HYBRID-PUR', areaInSft: 14.08, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73293-330-08-FG', description: 'BREAKER MOUNTING PAN-PUR-1', areaInSft: 0.79, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-010-51-FG', description: 'ENS. CAJA MP. 2 POS 125 A', areaInSft: 17.96, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-010-51-PC', description: 'ENS. CAJA MP. 2 POS 125 A', areaInSft: 17.96, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-011-52-FG', description: 'ENS. CAJA MP. 4 POS 125A', areaInSft: 31.38, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-011-52-PC', description: 'ENS. CAJA MP. 4 POS 125A', areaInSft: 31.38, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-012-52-FG', description: 'ENS. CAJA MP 6 POSICIONES 125A', areaInSft: 38.11, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-012-52-PC', description: 'ENS. CAJA MP 6 POSICIONES 125A', areaInSft: 38.11, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-022-53-FG', description: 'ENS. CAJA MP 2 3 POS. 200A-PUR', areaInSft: 35.24, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-023-51-FG', description: 'ENS. CAJA MP. 4 POS 200 A', areaInSft: 40.24, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-023-51-PC', description: 'ENS. CAJA MP. 4 POS 200 A', areaInSft: 40.24, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-031-51-FG', description: 'ENS. CAJA MP. 5-6 POS 200 A', areaInSft: 49.91, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-031-51-PC', description: 'ENS. CAJA MP. 5-6 POS 200 A', areaInSft: 49.91, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-032-65-FG', description: 'ENS CAJA MAIN 1 EUSERC', areaInSft: 48.27, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-032-74-FG', description: 'ENS CAJA MAIN', areaInSft: 37.46, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-213-51-FG', description: 'ENS. BARRERA MP 4 POS. 125A', areaInSft: 5.06, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-213-51-PC', description: 'ENS. BARRERA MP 4 POS. 125A', areaInSft: 5.06, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-214-50-FG', description: 'ENS. BARRERA ACOM. MP 4 POS.', areaInSft: 6.85, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-214-50-PC', description: 'ENS. BARRERA ACOM. MP 4 POS.', areaInSft: 6.85, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-215-51-FG', description: 'ENS. BARRERA DER. MP 6 POS.', areaInSft: 4.25, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-215-51-PC', description: 'ENS. BARRERA DER. MP 6 POS.', areaInSft: 4.25, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-216-51-FG', description: 'ENS. BARRERA IZQ. MP 6 POS.', areaInSft: 4.27, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-216-51-PC', description: 'ENS. BARRERA IZQ. MP 6 POS.', areaInSft: 4.27, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-217-50-FG', description: 'ENS. BARRERA IZQ. ACOM. MP 6', areaInSft: 6.04, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-217-50-PC', description: 'ENS. BARRERA IZQ. ACOM. MP 6', areaInSft: 6.04, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-218-50-FG', description: 'ENS. BARRERA DER. ACOM. MP 6', areaInSft: 5.98, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-218-50-PC', description: 'ENS. BARRERA DER. ACOM. MP 6', areaInSft: 5.98, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-227-50-FG', description: 'ENS. BARRERA MP 2,3 P 200A', areaInSft: 4.75, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-227-50-PC', description: 'ENS. BARRERA MP 2,3 P 200A', areaInSft: 4.75, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-230-50-FG', description: 'ENS. BARRERA MP 4 POS. 200A', areaInSft: 5.63, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-230-50-PC', description: 'ENS. BARRERA MP 4 POS. 200A', areaInSft: 5.63, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-232-50-FG', description: 'ENS. BARRERA ACOM.MP 4 POS.200', areaInSft: 8.15, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-232-50-PC', description: 'ENS. BARRERA ACOM.MP 4 POS.200', areaInSft: 8.15, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-235-50-FG', description: 'ENS. BARRERA MP 5-6 POS 200A', areaInSft: 4.74, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-235-50-PC', description: 'ENS. BARRERA MP 5-6 POS 200A', areaInSft: 4.74, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-316-50-FG', description: 'ENS. CUB. INTS. MP 4 POS. 125A', areaInSft: 5.33, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-316-50-PC', description: 'ENS. CUB. INTS. MP 4 POS. 125A', areaInSft: 5.33, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-317-50-FG', description: 'ENS. CUB. INTS. MP 6 POS. 125A', areaInSft: 5.17, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-317-50-PC', description: 'ENS. CUB. INTS. MP 6 POS. 125A', areaInSft: 5.17, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-324-02-FG', description: 'CUBIERTA ACOM. MP 2 3 5 6P 200-PUR', areaInSft: 8.07, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-361-01-FG', description: 'CUBIERTA SUP.EUSERC C.B.400A-PUR', areaInSft: 8.79, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-362-01-FG', description: 'CUBIERTA INF MCB EUSERC 400A-PUR', areaInSft: 11.93, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73296-537-02-FG', description: 'TAPA PARA X-BUS-PUR', areaInSft: 1.33, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73297-104-51-FG', description: 'CAJA ENS BRANCH 5 POS 125A', areaInSft: 23.17, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73297-114-51-FG', description: 'ENS CAJA BRANCH 3 POS 200A', areaInSft: 24.49, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73297-121-52-FG', description: 'CAJA ENS. BRANCH. COMER. 20S-PUR', areaInSft: 26.43, imageLink: 'http://localhost:3000/images/image.png' },
    { mainPartNo: '73297-153-51-FG', description: 'BERRERA ENS BRANCH 5 PS 125A', areaInSft: 6.29, imageLink: 'http://localhost:3000/images/image.png' }

];

async function main() {
    const surfaceAreaByHour: Record<number, number> = {};
    const productIds: Record<string, number> = {};
    const monthlySurfaceArea: Record<string, number> = {};
    const yearlySurfaceArea: Record<string, number> = {};
    const dailyProductRecords: Record<number, Record<string, number>> = {};

    // Fetch existing products and their IDs
    for (const product of products) {
        try {
            const existingProduct = await prisma.product.findFirst({
                where: {
                    mainPartNo: product.mainPartNo
                }
            });

            if (!existingProduct) {
                console.log(`Product with mainPartNo ${product.mainPartNo} does not exist.`);
                continue;
            }

            // Store the product ID in the map
            productIds[product.mainPartNo] = existingProduct.id;

            // Calculate the current hour
            const currentHour = new Date().getHours() - 1;
            const currentDay = new Date().getDate();
            const currentMonth = new Date().getMonth() + 1; // Months are 0-based in JS
            const currentYear = new Date().getFullYear();

            // Accumulate the surface area for the current hour
            if (surfaceAreaByHour[currentHour]) {
                surfaceAreaByHour[currentHour] += product.areaInSft;
            } else {
                surfaceAreaByHour[currentHour] = product.areaInSft;
            }

            // Accumulate the surface area for the current day, month, and year
            if (!dailyProductRecords[currentDay]) {
                dailyProductRecords[currentDay] = {};
            }
            if (dailyProductRecords[currentDay][product.mainPartNo]) {
                dailyProductRecords[currentDay][product.mainPartNo] += 1;
            } else {
                dailyProductRecords[currentDay][product.mainPartNo] = 1;
            }

            const productArea = product.areaInSft;

            // Monthly and yearly aggregates
            monthlySurfaceArea[`${currentMonth}-${currentYear}`] = (monthlySurfaceArea[`${currentMonth}-${currentYear}`] || 0) + productArea;
            yearlySurfaceArea[currentYear] = (yearlySurfaceArea[currentYear] || 0) + productArea;
        } catch (error) {
            console.error(`Error fetching product ${product.mainPartNo}:`, error);
        }
    }

    // Insert aggregated records into DailyRecord
    for (const [hour, totalSurfaceArea] of Object.entries(surfaceAreaByHour)) {
        try {
            await prisma.dailyRecord.create({
                data: {
                    totalSurfaceAreaProduced: totalSurfaceArea,
                    day: new Date().getDate(),
                    month: new Date().getMonth() + 1,
                    year: new Date().getFullYear(),
                    hour: parseInt(hour),
                    dailyProductRecords: {
                        create: Object.entries(productIds).map(([mainPartNo, id]) => ({
                            productId: id,
                            partNumber: mainPartNo,
                            count: dailyProductRecords[new Date().getDate()][mainPartNo] || 0
                        }))
                    }
                }
            });
        } catch (error) {
            console.error(`Error inserting daily record for hour ${hour}:`, error);
        }
    }

    // Insert aggregated records into MonthlyRecord
    for (const [monthYear, totalSurfaceArea] of Object.entries(monthlySurfaceArea)) {
        try {
            await prisma.monthlyRecord.create({
                data: {
                    totalSurfaceAreaProduced: totalSurfaceArea + 20,
                    day: 2, // Aggregated data does not have a specific day
                    month: parseInt(monthYear.split('-')[0]),
                    year: parseInt(monthYear.split('-')[1]),
                    monthlyProductRecords: {
                        create: Object.entries(dailyProductRecords).flatMap(([day, products]) => {
                            return Object.entries(products).map(([partNumber, count]) => {
                                return {
                                    productId: productIds[partNumber],
                                    partNumber,
                                    count
                                };
                            });
                        })
                    }
                }
            });
        } catch (error) {
            console.error(`Error inserting monthly record for ${monthYear}:`, error);
        }
    }

    // Insert aggregated records into YearlyRecord
    for (const [year, totalSurfaceArea] of Object.entries(yearlySurfaceArea)) {
        try {
            await prisma.yearlyRecord.create({
                data: {
                    totalSurfaceAreaProduced: totalSurfaceArea + 10,
                    year: parseInt(year),
                    month: 1, // Aggregated data does not have a specific month
                    yearlyProductRecords: {
                        create: Object.entries(dailyProductRecords).flatMap(([day, products]) => {
                            return Object.entries(products).map(([partNumber, count]) => {
                                return {
                                    productId: productIds[partNumber],
                                    partNumber,
                                    count
                                };
                            });
                        })
                    }
                }
            });
        } catch (error) {
            console.error(`Error inserting yearly record for ${year}:`, error);
        }
    }
}

main()
    .catch(e => {
        console.error(e);
        throw e;
    })
    .finally(async () => {
        await prisma.$disconnect();
    });
// const products = [
//     { mainPartNo: '40515-926-01-FG', description: 'COVER-PUR', areaInSft: 1.71, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-011-51-FG', description: 'ENS CAJA TAP BOX 1200-1600A', areaInSft: 45.37, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-030-13-FG', description: 'BOX BLANK 1200A EUSERC', areaInSft: 29.35, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-094-52-FG', description: 'ASSY BOTTOM EUSERC 1PH HYBRID-PUR', areaInSft: 3.81, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-094-01-FG', description: 'RESPALDO EUSERC 3PH HYBRID -PUR', areaInSft: 24.67, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-094-52-FG', description: 'ASSY BOTTOM EUSERC 1PH HYBRID-PUR', areaInSft: 3.81, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-095-50-FG', description: 'ASSY BOTTOM EUSERC 3PH HYBRID', areaInSft: 4.93, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-098-02-FG', description: 'BOTTOM COVER FUSE 1PH HYBRID-PUR', areaInSft: 9.50, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-116-02-FG', description: 'BOTTOM COVER BKR 3PH HYBRID-PUR', areaInSft: 14.08, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73293-330-08-FG', description: 'BREAKER MOUNTING PAN-PUR-1', areaInSft: 0.79, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-010-51-FG', description: 'ENS. CAJA MP. 2 POS 125 A', areaInSft: 17.96, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-010-51-PC', description: 'ENS. CAJA MP. 2 POS 125 A', areaInSft: 17.96, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-011-52-FG', description: 'ENS. CAJA MP. 4 POS 125A', areaInSft: 31.38, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-011-52-PC', description: 'ENS. CAJA MP. 4 POS 125A', areaInSft: 31.38, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-012-52-FG', description: 'ENS. CAJA MP 6 POSICIONES 125A', areaInSft: 38.11, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-012-52-PC', description: 'ENS. CAJA MP 6 POSICIONES 125A', areaInSft: 38.11, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-022-53-FG', description: 'ENS. CAJA MP 2 3 POS. 200A-PUR', areaInSft: 35.24, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-023-51-FG', description: 'ENS. CAJA MP. 4 POS 200 A', areaInSft: 40.24, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-023-51-PC', description: 'ENS. CAJA MP. 4 POS 200 A', areaInSft: 40.24, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-031-51-FG', description: 'ENS. CAJA MP. 5-6 POS 200 A', areaInSft: 49.91, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-031-51-PC', description: 'ENS. CAJA MP. 5-6 POS 200 A', areaInSft: 49.91, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-032-65-FG', description: 'ENS CAJA MAIN 1 EUSERC', areaInSft: 48.27, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-032-74-FG', description: 'ENS CAJA MAIN', areaInSft: 37.46, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-213-51-FG', description: 'ENS. BARRERA MP 4 POS. 125A', areaInSft: 5.06, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-213-51-PC', description: 'ENS. BARRERA MP 4 POS. 125A', areaInSft: 5.06, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-214-50-FG', description: 'ENS. BARRERA ACOM. MP 4 POS.', areaInSft: 6.85, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-214-50-PC', description: 'ENS. BARRERA ACOM. MP 4 POS.', areaInSft: 6.85, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-215-51-FG', description: 'ENS. BARRERA DER. MP 6 POS.', areaInSft: 4.25, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-215-51-PC', description: 'ENS. BARRERA DER. MP 6 POS.', areaInSft: 4.25, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-216-51-FG', description: 'ENS. BARRERA IZQ. MP 6 POS.', areaInSft: 4.27, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-216-51-PC', description: 'ENS. BARRERA IZQ. MP 6 POS.', areaInSft: 4.27, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-217-50-FG', description: 'ENS. BARRERA IZQ. ACOM. MP 6', areaInSft: 6.04, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-217-50-PC', description: 'ENS. BARRERA IZQ. ACOM. MP 6', areaInSft: 6.04, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-218-50-FG', description: 'ENS. BARRERA DER. ACOM. MP 6', areaInSft: 5.98, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-218-50-PC', description: 'ENS. BARRERA DER. ACOM. MP 6', areaInSft: 5.98, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-227-50-FG', description: 'ENS. BARRERA MP 2,3 P 200A', areaInSft: 4.75, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-227-50-PC', description: 'ENS. BARRERA MP 2,3 P 200A', areaInSft: 4.75, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-230-50-FG', description: 'ENS. BARRERA MP 4 POS. 200A', areaInSft: 5.63, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-230-50-PC', description: 'ENS. BARRERA MP 4 POS. 200A', areaInSft: 5.63, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-232-50-FG', description: 'ENS. BARRERA ACOM.MP 4 POS.200', areaInSft: 8.15, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-232-50-PC', description: 'ENS. BARRERA ACOM.MP 4 POS.200', areaInSft: 8.15, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-235-50-FG', description: 'ENS. BARRERA MP 5-6 POS 200A', areaInSft: 4.74, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-235-50-PC', description: 'ENS. BARRERA MP 5-6 POS 200A', areaInSft: 4.74, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-316-50-FG', description: 'ENS. CUB. INTS. MP 4 POS. 125A', areaInSft: 5.33, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-316-50-PC', description: 'ENS. CUB. INTS. MP 4 POS. 125A', areaInSft: 5.33, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-317-50-FG', description: 'ENS. CUB. INTS. MP 6 POS. 125A', areaInSft: 5.17, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-317-50-PC', description: 'ENS. CUB. INTS. MP 6 POS. 125A', areaInSft: 5.17, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-324-02-FG', description: 'CUBIERTA ACOM. MP 2 3 5 6P 200-PUR', areaInSft: 8.07, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-361-01-FG', description: 'CUBIERTA SUP.EUSERC C.B.400A-PUR', areaInSft: 8.79, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-362-01-FG', description: 'CUBIERTA INF MCB EUSERC 400A-PUR', areaInSft: 11.93, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73296-537-02-FG', description: 'TAPA PARA X-BUS-PUR', areaInSft: 1.33, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-104-51-FG', description: 'CAJA ENS BRANCH 5 POS 125A', areaInSft: 23.17, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-114-51-FG', description: 'ENS CAJA BRANCH 3 POS 200A', areaInSft: 24.49, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-121-52-FG', description: 'CAJA ENS. BRANCH. COMER. 20S-PUR', areaInSft: 26.43, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-153-51-FG', description: 'BERRERA ENS BRANCH 5 PS 125A', areaInSft: 6.29, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-156-50-FG', description: 'BERRERA ENS BRANCH 4 P 200A', areaInSft: 5.59, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-158-50-FG', description: 'BARRERA ENS BRANCH 3 POS 200A', areaInSft: 4.79, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-202-50-FG', description: 'ENS CUB INT BRANCH 5 POS 125A', areaInSft: 6.08, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: '73297-229-50-FG', description: 'ENS CUB INT BRANCH RES 3P 200A', areaInSft: 7.87, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: 'MTR11751-FG', description: 'ERMS BRACKET 1600-2000-PUR', areaInSft: 1.89, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: 'MTR12000-FG', description: 'BOX BLANK 1000A EUSERC M02 -PUR', areaInSft: 29.35, imageLink: 'http://localhost:3000/images/image.png' },
//     { mainPartNo: 'MTR12003-P', description: 'ASSY LEFT PANEL 1000A EUSERC-PUR', areaInSft: 13.26, imageLink: 'http://localhost:3000/images/image.png' },
// ];

