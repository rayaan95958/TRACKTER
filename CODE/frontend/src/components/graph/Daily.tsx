import React, { useState, useEffect } from 'react';
import { ResponsiveLine } from '@nivo/line';
import { FaRegImage } from 'react-icons/fa';

interface DailyProductRecord {
    productId: number;
    partNumber: string;
    count: number;
}

interface DailyRecord {
    totalSurfaceAreaProduced: number;
    day: number;
    month: number;
    year: number;
    hour: number;
    dailyProductRecords: DailyProductRecord[];
}

interface Product {
    mainPartNo: string;
    description: string;
    imageLink: string;
}

interface DailyProps {
    selectedDate: Date;
}

const Daily: React.FC<DailyProps> = ({ selectedDate }) => {
    const [data, setData] = useState<DailyRecord[]>([]);
    const [products, setProducts] = useState<{ [key: string]: Product }>({});
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            const day = selectedDate.getDate();
            const month = selectedDate.getMonth() + 1; // getMonth() is zero-based
            const year = selectedDate.getFullYear();

            try {
                // Fetch daily records
                const response = await fetch(`/api/daily?day=${day}&month=${month}&year=${year}`);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setData(data);

                // Fetch products
                const productsResponse = await fetch('/api/products');
                if (!productsResponse.ok) {
                    throw new Error('Network response was not ok');
                }
                const productsData = await productsResponse.json();
                const productsMap = productsData.reduce((acc: { [key: string]: Product }, product: Product) => {
                    acc[product.mainPartNo] = product;
                    return acc;
                }, {});
                setProducts(productsMap);
            } catch (error) {
                let errorMessage = "Failed to fetch data";
                if (error instanceof Error) {
                    errorMessage = error.message;
                }
                setError(errorMessage);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [selectedDate]);

    // Calculate total surface area
    const totalSurfaceArea = data.reduce((sum, record) => sum + record.totalSurfaceAreaProduced, 0);

    // Aggregate product counts
    const productCounts: { [key: string]: { description: string; count: number; imageLink: string } } = {};
    data.forEach(record => {
        record.dailyProductRecords.forEach(productRecord => {
            if (products[productRecord.partNumber]) {
                const product = products[productRecord.partNumber];
                if (productCounts[productRecord.partNumber]) {
                    productCounts[productRecord.partNumber].count += productRecord.count;
                } else {
                    productCounts[productRecord.partNumber] = {
                        description: product.description,
                        count: productRecord.count,
                        imageLink: product.imageLink,
                    };
                }
            }
        });
    });

    // Prepare data for the line chart
    const chartData = [
        {
            id: 'Surface Area Produced',
            data: data.map(record => ({
                x: `${record.hour}:00`,
                y: record.totalSurfaceAreaProduced,
            })),
        },
    ];

    const chartTheme = {
        axis: {
            domain: {
                line: {
                    stroke: '#ffffff',
                },
            },
            legend: {
                text: {
                    fill: '#ffffff',
                },
            },
            ticks: {
                text: {
                    fill: '#ffffff',
                },
            },
        },
        legends: {
            text: {
                fill: '#ffffff',
            },
        },
        tooltip: {
            container: {
                color: '#ffffff',
            },
        },
    };

    const formattedDate = `${selectedDate.getDate().toString().padStart(2, '0')}/${(selectedDate.getMonth() + 1).toString().padStart(2, '0')}/${selectedDate.getFullYear()}`;

    return (
        <div className="w-full">
            <h1 className="text-2xl font-bold mb-2 text-center">DAILY SURFACE AREA COATED</h1>
            <p className="text-lg text-center mb-4">Graph for {formattedDate}</p>
            {loading && <p className="text-center">Loading...</p>}
            {error && <p className="text-red-500 text-center">Error: {error}</p>}
            {!loading && !error && data.length === 0 && (
                <p className="text-center">No data available</p>
            )}
            {!loading && !error && data.length > 0 && (
                <div className="flex flex-col items-center">
                    <div className="w-full max-w-6xl mx-auto mb-4 flex justify-center items-center">
                        <div className="h-96 w-full">
                            <ResponsiveLine
                                data={chartData}
                                margin={{ top: 50, right: 110, bottom: 50, left: 70 }}
                                xScale={{ type: 'point' }}
                                yScale={{
                                    type: 'linear',
                                    min: 'auto',
                                    max: 'auto',
                                    stacked: false,
                                    reverse: false,
                                }}
                                curve="linear"
                                axisTop={null}
                                axisRight={null}
                                axisBottom={{
                                    tickSize: 5,
                                    tickPadding: 5,
                                    tickRotation: 0,
                                    legend: 'Hour of the Day',
                                    legendOffset: 36,
                                    legendPosition: 'middle',
                                }}
                                axisLeft={{
                                    tickSize: 5,
                                    tickPadding: 5,
                                    tickRotation: 0,
                                    legend: 'Surface Area Produced (sq. ft)',
                                    legendOffset: -40,
                                    legendPosition: 'middle',
                                }}
                                pointSize={10}
                                pointColor={{ from: 'color', modifiers: [] }}
                                pointBorderWidth={2}
                                pointBorderColor={{ from: 'serieColor' }}
                                pointLabelYOffset={-12}
                                enableCrosshair={true}
                                enableGridX={false} // Disable vertical grid lines
                                enableGridY={false} // Disable horizontal grid lines
                                colors={['rgba(255, 191, 0,0.5)']}
                                lineWidth={3}
                                enablePoints={true}
                                enableArea={false}
                                enableSlices="x"
                                sliceTooltip={({ slice }) => {
                                    const total = slice.points.reduce((sum, point) => sum + (typeof point.data.y === 'number' ? point.data.y : 0), 0);
                                    return (
                                        <div style={{ padding: '12px', background: 'rgba(0, 0, 0, 0.7)', color: '#ffffff' }}>
                                            <div>Time/Date: {slice.points[0].data.xFormatted}, {formattedDate}</div>
                                            <div>Total Surface Area: {total.toFixed(2)} sq. ft</div>
                                        </div>
                                    );
                                }}
                                theme={chartTheme}
                            />
                        </div>
                    </div>

                    <p className="text-lg font-semibold text-center">Total Surface Area Coated: {totalSurfaceArea} sq. ft</p>
                    <hr className="border-t-4 border-gray-300 w-full mb-8 mt-12" />
                    <div className="mt-8 w-full max-w-6xl px-4 sm:px-6 md:px-8">
                        <h2 className="text-2xl font-bold mb-8 text-center">PRODUCTS COATED</h2>
                        <div className="overflow-x-auto">
                            <table className="min-w-full bg-black text-white text-xs sm:text-sm md:text-base lg:text-lg">
                                <thead>
                                    <tr className="border-b border-gray-700">
                                        <th className="py-2 px-3 sm:px-4 border-b text-center">Part Number</th>
                                        <th className="py-2 px-3 sm:px-4 border-b text-center">Image</th>
                                        <th className="py-2 px-3 sm:px-4 border-b text-center">Description</th>
                                        <th className="py-2 px-3 sm:px-4 border-b text-center">Quantity</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.keys(productCounts).map(partNo => (
                                        <tr key={partNo} className="hover:bg-gray-800">
                                            <td className="py-2 px-3 sm:px-4 border-b text-center">{partNo}</td>
                                            <td className="py-2 px-3 sm:px-4 border-b text-center">
                                                <a href={productCounts[partNo].imageLink} target="_blank" rel="noopener noreferrer">
                                                    <FaRegImage className="inline-block text-xl" />
                                                </a>
                                            </td>
                                            <td className="py-2 px-3 sm:px-4 border-b text-center">{productCounts[partNo].description}</td>
                                            <td className="py-2 px-3 sm:px-4 border-b text-center">{productCounts[partNo].count}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Daily;
