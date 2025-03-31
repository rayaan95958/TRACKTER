"use client";
import React, { useState, useEffect } from 'react';
import { FaRegImage } from 'react-icons/fa';
import YellowButton from "../../../components/button/YellowButton";
import GreyButton from "../../../components/button/GreyButton";

// Define the type for product
interface Product {
  id: number;
  mainPartNo: string;
  imageLink: string;
  description: string;
  areaInSft: number;
}

export default function Database() {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchProducts = async () => {
      try {
        const response = await fetch(`/api/products`);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data: Product[] = await response.json();
        setProducts(data);
      } catch (error) {
        if (error instanceof Error) {
          setError(error.message);
        } else {
          setError('An unknown error occurred');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchProducts();
  }, []);

  return (
    <>
      <div className="flex justify-center mt-10 md:mt-12 lg:mt-16 gap-4 lg:gap-8">
        <YellowButton text="Database" href="/database" />
        <GreyButton text="Main Page" href="/" />
      </div>
      <div className="container mx-auto mt-10 px-4 sm:px-0">
        {loading && <p className="text-center text-gray-500 text-sm md:text-base">Loading...</p>}
        {error && <p className="text-center text-red-500 text-sm md:text-base">Error: {error}</p>}
        {!loading && !error && (
          <div className="overflow-x-auto w-full">
            <table className="min-w-full bg-black text-white text-xs md:text-sm lg:text-base">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="py-1 px-2 sm:py-2 sm:px-4 text-center">ID</th>
                  <th className="py-1 px-2 sm:py-2 sm:px-4 text-center">Main Part No</th>
                  <th className="py-1 px-2 sm:py-2 sm:px-4 text-center">Image</th>
                  <th className="py-1 px-2 sm:py-2 sm:px-4 text-center">Description</th>
                  <th className="py-1 px-2 sm:py-2 sm:px-4 text-center">Area (sq. ft)</th>
                </tr>
              </thead>
              <tbody>
                {products.map((product) => (
                  <tr key={product.id} className="border-t border-gray-700 hover:bg-gray-800">
                    <td className="py-1 px-2 sm:py-2 sm:px-4 text-center">{product.id}</td>
                    <td className="py-1 px-2 sm:py-2 sm:px-4 text-center">{product.mainPartNo}</td>
                    <td className="py-1 px-2 sm:py-2 sm:px-4 text-center">
                      <a href={product.imageLink} target="_blank" rel="noopener noreferrer" className="flex justify-center">
                        <FaRegImage className="text-white text-lg sm:text-xl md:text-2xl" />
                      </a>
                    </td>
                    <td className="py-1 px-2 sm:py-2 sm:px-4 text-center">{product.description}</td>
                    <td className="py-1 px-2 sm:py-2 sm:px-4 text-center">{product.areaInSft}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
