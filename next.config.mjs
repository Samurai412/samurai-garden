import { createRequire } from 'module';
const require = createRequire(import.meta.url);

const withMDX = require('@next/mdx')({
  extension: /\.mdx?$/,
});

export default withMDX({
  pageExtensions: ['js', 'jsx', 'md', 'mdx'],
  eslint: {
    ignoreDuringBuilds: true,
  },
  webpack: (config) => {
    config.infrastructureLogging = {
      level: 'error',
    };
    return config;
  },
});
