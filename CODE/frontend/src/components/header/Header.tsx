'use client';

import Link from 'next/link';
import ThemeSwitcher from '../theme/ThemeSwitcher';

interface HeaderProps {
  isDarkMode: boolean;
}

const Header = ({isDarkMode}: HeaderProps) => {

  return (
    <header
      className={`${
        isDarkMode ? 'w-full border-b bg-gray-800 border-gray-700' : 'w-full border-b bg-gray-800 border-gray-200'
      }`}
    >
      <nav className="flex flex-col lg:flex-row lg:justify-between p-2 lg:p-4">
        <div className="container mx-auto flex items-center justify-between p-2 lg:p-3">
          <div className="flex items-center gap-2 md:gap-3 lg:gap-4">
            <img
              src="./icons/icon-512x512.png"
              alt="Logo"
              className="h-14 md:h-16 lg:h-20 w-auto"
            />
            <h1 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold">
              <Link href="/" className="hover:text-gray-300">
                Trackter
              </Link>
            </h1>
          </div>
          <div className="flex items-center gap-2 md:gap-3 lg:gap-4">
            <ThemeSwitcher />
          </div>
        </div>
      </nav>
    </header>
  );
};

export default Header;