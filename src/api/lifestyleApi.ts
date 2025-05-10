// import axiosInstance from './axiosInstance';
// import { loadTokens } from '../redux/actions/authAction.ts';

export const saveLifestyle = async (questionList: { question_id: number; answer: string }[]) => {
  // FOR LOCAL TEST
  if (questionList[0].answer === '1인') {
    return { statusCode: 200, message: '라이프스타일 저장 성공' };
  } else if (questionList[0].answer === '2인') {
    return { statusCode: 400, message: '라이프스타일 저장 실패' };
  } else {
    return { statusCode: 401, message: '인증 실패' };
  }

  // FOR SERVER COMMUNICATION
  // const { accessToken } = await loadTokens();
  // const response = await axiosInstance.post(
  //   '/lifestyle/save',
  //   { question_list: questionList },
  //   {
  //     headers: {
  //       Authorization: `Bearer ${accessToken}`,
  //     },
  //   },
  // );
  //
  // return response.data;
};