import axiosInstance from './axiosInstance';
import { loadTokens } from '../redux/actions/authAction.ts';

export const saveLifestyle = async (questionList: { question_id: number; answer: string }[]) => {
  // FOR LOCAL TEST
  // if (questionList[0].answer === '1') {
    // return { statusCode: 200, message: '라이프스타일 저장 성공' };
  // } else if (questionList[0].answer === '2') {
  //   return { statusCode: 400, message: '라이프스타일 저장 실패' };
  // } else {
  //   return { statusCode: 401, message: '인증 실패' };
  // }

  // FOR SERVER COMMUNICATION
  const { accessToken } = await loadTokens();

  console.log('*라이프스타일 저장 api request body:', { question_list: questionList });

  try {
    const response = await axiosInstance.post(
      '/lifestyle/save',
      { question_list: questionList },
      {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      },
    );

    // if (!response.ok) {
    //   console.log('라이프스타일 조회 실패');
    //   console.log(response);
    // }

    console.log('라이프스타일 저장 성공');
    console.log('*라이프스타일 저장 api response:', response.data);

    return response.data;
  } catch (error) {
    console.log('라이프스타일 저장 실패');
    console.log(error);

    // return rejectWithValue('알 수 없는 오류가 발생했습니다.');
  }
};