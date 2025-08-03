import withPWAInit from "@ducanh2912/next-pwa";

const withPWA = withPWAInit({
  output: 'standalone',
  pwa: {
    dest: 'public',
    register: true,
    skipWaiting: true,
    disable: process.env.NODE_ENV === 'development',
  }
});

export default withPWA({
  // Your Next.js config
});