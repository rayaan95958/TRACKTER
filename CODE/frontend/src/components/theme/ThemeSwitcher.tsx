'use client';

import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';
import { CiSun } from 'react-icons/ci';
import { IoMoonOutline } from 'react-icons/io5';

const ThemeSwitcher = () => {
  const { theme, setTheme, systemTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  const currentTheme = theme === 'system' ? systemTheme : theme;
  const isDark = currentTheme === 'dark';

  return (
    <button
      onClick={() => setTheme(isDark ? 'light' : 'dark')}
      className={`flex items-center justify-center gap-2 
        text-sm py-2 px-4 
        sm:text-base sm:py-3 sm:px-6 
        md:text-lg md:py-2 md:px-6 
        lg:text-xl lg:py-3 lg:px-7 
        rounded-lg shadow-lg font-semibold 
        transition-all duration-300 ease-in-out 
        active:border-gray-300 active:shadow-md
        ${isDark
          ? 'bg-white text-black hover:bg-gray-200 hover:text-black'
          : 'bg-black text-white hover:bg-gray-600 hover:text-white'}
      `}
      aria-label="Toggle Theme"
    >
      {isDark ? (
        <>
          <CiSun className="text-xl sm:text-2xl" />
          Light Mode
        </>
      ) : (
        <>
          <IoMoonOutline className="text-xl sm:text-2xl" />
          Dark Mode
        </>
      )}
    </button>
  );
};

export default ThemeSwitcher;