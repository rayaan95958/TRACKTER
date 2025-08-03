import './globals.css';
import { DM_Sans } from 'next/font/google';
import { ThemeProvider } from './theme-provider';

const dm = DM_Sans({ subsets: ['latin'] });

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head />
      <body className={`${dm.className}`}>
      <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      {children}
      </ThemeProvider>
      </body>
    </html>
  );
}