import type { NextApiRequest, NextApiResponse } from 'next';
import prisma from '../../lib/prisma';

export default async function handle(req: NextApiRequest, res: NextApiResponse) {
    const { month, year } = req.query;

    if (!month || !year) {
        return res.status(400).json({ error: 'month and year are required.' });
    }

    try {
        const records = await prisma.monthlyRecord.findMany({
            where: {
                month: parseInt(month as string, 10),
                year: parseInt(year as string, 10),
            },
            include: {
                monthlyProductRecords: true,
            },
        });
        return res.status(200).json(records);
    } catch (error) {
        console.error('Error fetching records:', error);
        return res.status(500).json({ error: 'Internal Server Error' });
    }
}
