import React from 'react';
import { FaUser, FaEnvelope, FaPhone } from 'react-icons/fa';

const Contact: React.FC = () => {
    return (
        <div id="contact" className="w-full min-h-screen flex flex-col items-center justify-center bg-black p-4">
            <h1 className="text-lg sm:text-xl md:text-2xl lg:text-3xl font-bold mb-6 text-center text-white">
                Contact Information
            </h1>
            <div className="flex flex-col space-y-4">
                <div className="flex flex-col md:flex-row md:space-x-4 items-center">
                    <div className="flex items-center space-x-2 md:space-x-4">
                        <FaUser className="text-white text-lg md:text-xl" />
                        <span className="text-white text-sm md:text-base">Samyak Singhal</span>
                    </div>
                    <div className="flex items-center space-x-2 md:space-x-4">
                        <FaEnvelope className="text-white text-lg md:text-xl" />
                        <span className="text-white text-sm md:text-base">samyaksinghal2003@gmail.com</span>
                    </div>
                    <div className="flex items-center space-x-2 md:space-x-4">
                        <FaPhone className="text-white text-lg md:text-xl" />
                        <span className="text-white text-sm md:text-base">+91 8105695390</span>
                    </div>
                </div>
                <div className="flex flex-col md:flex-row md:space-x-4 items-center">
                    <div className="flex items-center space-x-2 md:space-x-4">
                        <FaUser className="text-white text-lg md:text-xl" />
                        <span className="text-white text-sm md:text-base">Rayaan Sattar</span>
                    </div>
                    <div className="flex items-center space-x-2 md:space-x-4">
                        <FaEnvelope className="text-white text-lg md:text-xl" />
                        <span className="text-white text-sm md:text-base">satarrayan@gmail.com</span>
                    </div>
                    <div className="flex items-center space-x-2 md:space-x-4">
                        <FaPhone className="text-white text-lg md:text-xl" />
                        <span className="text-white text-sm md:text-base">+91 7204679328</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Contact;
