import withMDX from '@next/mdx';

const withMDXConfig = withMDX({
  extension: /\.mdx?$/,
});

export default {
  ...withMDXConfig,
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
};
