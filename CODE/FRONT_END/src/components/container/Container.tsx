"use client";
import React, { useState } from 'react';
import Calender from "../calender/Calender";
import Daily from "../graph/Daily";
import Monthly from "../graph/Monthly";
import Yearly from "../graph/Yearly";
import { SlCalender } from "react-icons/sl";

const Container: React.FC = () => {
  const [isCalenderVisible, setIsCalenderVisible] = useState(false);
  const [visibleComponent, setVisibleComponent] = useState('Daily');
  const [selectedDate, setSelectedDate] = useState(new Date());

  const toggleCalender = () => {
    setIsCalenderVisible(!isCalenderVisible);
  };

  const handleDateSelect = (day: number, month: number, year: number) => {
    setSelectedDate(new Date(year, month - 1, day));
  };

  const renderComponent = () => {
    switch (visibleComponent) {
      case 'Daily':
        return <Daily selectedDate={selectedDate} />;
      case 'Monthly':
        return <Monthly selectedDate={selectedDate} />;
      case 'Yearly':
        return <Yearly selectedDate={selectedDate} />;
      default:
        return <Daily selectedDate={selectedDate} />;
    }
  };

  return (
    <div className="mx-auto w-4/5 mt-20 border border-gray-300 relative mb-16 pb-16">
      <div className="grid grid-cols-3">
        <button
          className="bg-gray-300 text-black text-xl border-none cursor-pointer transition duration-300 ease-in-out hover:bg-gray-700 hover:text-white"
          onClick={() => setVisibleComponent('Daily')}
        >
          Daily
        </button>
        <button
          className="bg-gray-300 py-2 text-black text-xl border-none cursor-pointer transition duration-300 ease-in-out hover:bg-gray-700 hover:text-white"
          onClick={() => setVisibleComponent('Monthly')}
        >
          Monthly
        </button>
        <button
          className="bg-gray-300 text-black text-xl border-none cursor-pointer transition duration-300 ease-in-out hover:bg-gray-700 hover:text-white"
          onClick={() => setVisibleComponent('Yearly')}
        >
          Yearly
        </button>
      </div>
      <button
        className="flex items-center bg-[#FFBF00] text-black py-2 px-4 border-none cursor-pointer transition duration-300 ease-in-out hover:bg-[#d0a00e]"
        onClick={toggleCalender}
      >
        <SlCalender className="w-7 h-7 mr-2" />
        Calendar
      </button>
      <div className={`mb-4 overflow-hidden transition-all duration-1000 ${isCalenderVisible ? 'max-h-72' : 'max-h-0'}`}>
        <Calender onDateSelect={handleDateSelect} />
      </div>
      <div className="transition transform duration-500 ease-in-out opacity-0 translate-y-5" style={{ opacity: visibleComponent ? 1 : 0, transform: visibleComponent ? 'translateY(0)' : 'translateY(-20px)' }}>
        {renderComponent()}
      </div>
    </div>
  );
};

export default Container;
