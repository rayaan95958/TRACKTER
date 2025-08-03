import type { NextApiRequest, NextApiResponse } from 'next';
import prisma from '../../lib/prisma';

export default async function handle(req: NextApiRequest, res: NextApiResponse) {
  const { day, month, year } = req.query;

  if (!day || !month || !year) {
    return res.status(400).json({ error: 'Day, month, and year are required.' });
  }

  try {
    const records = await prisma.dailyRecord.findMany({
      where: {
        day: parseInt(day as string, 10),
        month: parseInt(month as string, 10),
        year: parseInt(year as string, 10),
      },
      include: {
        dailyProductRecords: true,
      },
    });
    return res.status(200).json(records);
  } catch (error) {
    console.error('Error fetching records:', error);
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}
