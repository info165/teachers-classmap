import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  
  return {
define: {
  'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY || env.API_KEY || 'AIzaSyB-ZfFS49ofq8gb_MVIL807_bwv9BGGRRg'),
  'process.env.API_KEY': JSON.stringify(env.API_KEY || env.GEMINI_API_KEY || 'AIzaSyB-ZfFS49ofq8gb_MVIL807_bwv9BGGRRg'),
},
    build: {
      rollupOptions: {
        external: ['marked', '@google/genai'],
        output: {
          globals: {
            'marked': 'marked',
            '@google/genai': 'GoogleGenAI'
          }
        }
      }
    }
  };
});
