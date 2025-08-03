import React, { useState, useEffect } from 'react';
import Calendar, { CalendarProps } from 'react-calendar';
import './Calender.css';

type ValuePiece = Date | null;
type Value = ValuePiece;

interface CalenderProps {
    onDateSelect: (day: number, month: number, year: number) => void;
}

const Calender: React.FC<CalenderProps> = ({ onDateSelect }) => {
    const [value, setValue] = useState<Value>(new Date());

    useEffect(() => {
        if (value instanceof Date) {
            const day = value.getDate();
            const month = value.getMonth() + 1;
            const year = value.getFullYear();
            console.log(`Day: ${day}, Month: ${month}, Year: ${year}`);
            onDateSelect(day, month, year);
        }
    }, [value]);

    const handleDateChange: CalendarProps['onChange'] = (date) => {
        if (date instanceof Date) {
            setValue(date);
            const day = date.getDate();
            const month = date.getMonth() + 1;
            const year = date.getFullYear();
            console.log(`Day: ${day}, Month: ${month}, Year: ${year}`);
            onDateSelect(day, month, year);
        }
    };

    return (
        <Calendar onChange={handleDateChange} value={value} />
    );
};

export default Calender;
