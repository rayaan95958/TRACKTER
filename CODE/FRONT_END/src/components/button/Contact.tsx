import React from 'react';
import { FaPhoneAlt } from 'react-icons/fa';

const Contact = () => {
  return (
    <a href="/#contact">
      <div className="flex items-center rounded shadow-md bg-gray-300 text-gray-600 transition duration-300 ease-in-out hover:bg-gray-400 cursor-pointer px-3 py-2 md:px-8 md:py-4 lg:px-6 lg:py-3">
        <FaPhoneAlt className="mr-2 text-lg sm:text-xl md:text-lg lg:text-xl" />
        <h3 className="text-sm sm:text-base md:text-md lg:text-lg font-semibold">Contact Us</h3>
      </div>
    </a>
  );
};

export default Contact;