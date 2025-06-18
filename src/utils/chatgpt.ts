import axios from 'axios';
import { OPENAI_API_KEY } from '@env';

const API_KEY = OPENAI_API_KEY;

export const sendMessageToChatGPT = async (userMessage: string) => {
  try {
    const response = await axios.post(
      'https://api.openai.com/v1/chat/completions',
      {
        model: 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: '친절한 AI 비서입니다.' },
          { role: 'user', content: userMessage },
        ],
        temperature: 0.7,
      },
      {
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${API_KEY}`,
        },
      }
    );
    return response.data.choices[0].message.content;
  } catch (error) {
    console.error('ChatGPT 오류:', error);
    return '죄송합니다. 응답을 받을 수 없습니다.';
  }
};
