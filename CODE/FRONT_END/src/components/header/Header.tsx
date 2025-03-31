import Link from 'next/link';
import Contact from '../button/Contact';

const Header = () => {
  return (
    <nav className="flex flex-col lg:flex-row lg:justify-between p-2 lg:p-4 border-b border-gray-800">
      <div className="container mx-auto flex items-center justify-between p-2 lg:p-3">
        <div className="flex items-center gap-2 md:gap-3 lg:gap-4">
          <img
            src="./Logo.png"
            alt="Logo"
            className="h-14 md:h-16 lg:h-20 w-auto"
          />
          <h1 className="text-xl sm:text-2xl md:text-3xl lg:text-4xl font-bold">
            <Link href="/" className="text-white hover:text-gray-300">Trackter</Link>
          </h1>
        </div>
        <Contact />
      </div>
    </nav>
  );
};

export default Header;
