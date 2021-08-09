# pytorch-guide

> "PyTorch 가이드"크루의 목표는 Pytorch을 활용하여 인공지능의 기초적인 지식부터 가벼운 활용까지 다룰 수 있는 가짜연구소만의 탄탄한 자료를 구축하는 것입니다. 널리 활용되는 자료를 만들어 AI 생태계에 기여하고자 합니다.

- 노션 페이지 : [바로가기](https://www.notion.so/chanrankim/PyTorch-f56ab03f6ac1488bb108514d3eed9ab8)
- 빌드 페이지 : https://pseudo-lab.github.io/pytorch-guide/

## jupyter book build

- init
  - 페이지 작성은 `.md`, `.ipynb` 형식으로 작성
  - cmd 사용
    - anaconda prompt 설치 권장
    - 가상환경에서 설치 권장(옵션)

- git clone 

  - ```
    git clone https://github.com/Pseudo-Lab/pytorch-guide.git
    ```

- 페이지 작성 파일 이동

  - `pytorch-guide/book/docs` 에 위치시킬 것
  - `ch1` 폴더 내에 작성

- `_toc.yml` 변경

  - `pytorch-guide/book` 내 `_toc.yml` 파일 변경

  - ```yaml
    format: jb-book
    root: docs/index
    chapters:
    - file: docs/ch1/분류 (Classification)
      sections:
      - file: docs/ch1/CNN (Convolutional Neural Network)
    # - fime: docs/ch1/(작성한 파일 이름 작성)
    ```

  - 위 코드 참조하여 추가한 페이지 이름 변경

- Jupyter book 설치

  - ```
    pip install -U jupyter-book
    ```

- 폴더 이동

  - ```
    cd pytorch-guide
    ```

- (로컬) Jupyter book build

  - ```
    jupyter-book build book/
    ```

  - cmd 창 내 `Or paste this line directly into your browser bar:` 이하의 링크를 사용하면 로컬에서 jupyter book 을 빌드할 수 있음

- (온라인) Jupyter book build

  - 변경 내용 push 할 것

  - ```python
    pip install ghp-import
    ghp-import -n -p -f book/_build/html -m "20-08-09 publishing"
    ```

  - https://pseudo-lab.github.io/pytorch-guide/ 링크 접속

