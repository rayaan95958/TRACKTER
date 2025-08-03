import type { NextApiRequest, NextApiResponse } from 'next';
import prisma from '../../lib/prisma';

export default async function handle(req: NextApiRequest, res: NextApiResponse) {
    const { mainPartNo } = req.query;

    if (mainPartNo) {
        try {
            const prodRecords = await prisma.product.findFirst({
                where: {
                    mainPartNo: mainPartNo as string,
                },
            });

            if (!prodRecords) {
                return res.status(404).json({ error: 'Product not found' });
            }

            return res.status(200).json(prodRecords);
        } catch (error) {
            console.error('Error fetching records:', error);
            return res.status(500).json({ error: 'Internal Server Error' });
        }
    } else {
        try {
            const records = await prisma.product.findMany();
            return res.status(200).json(records);
        } catch (error) {
            console.error('Error fetching records:', error);
            return res.status(500).json({ error: 'Internal Server Error' });
        }
    }
}
