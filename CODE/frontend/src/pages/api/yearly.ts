import type { NextApiRequest, NextApiResponse } from 'next';
import prisma from '../../lib/prisma';

export default async function handle(req: NextApiRequest, res: NextApiResponse) {
    const { year } = req.query;

    if (!year) {
        return res.status(400).json({ error: 'year required.' });
    }

    try {
        const records = await prisma.yearlyRecord.findMany({
            where: {
                year: parseInt(year as string, 10),
            },
            include: {
                yearlyProductRecords: true,
            },
        });
        return res.status(200).json(records);
    } catch (error) {
        console.error('Error fetching records:', error);
        return res.status(500).json({ error: 'Internal Server Error' });
    }
}
