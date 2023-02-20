const ap = new APlayer({
    container: document.getElementById('aplayer'),
    fixed: true,
	autoplay: true, //自动播放
    audio: [
	{
        name: 'Hype Boy',
        artist: 'I',
        url: 'https://m704.music.126.net/20230220112146/32f18a83e517f9385085483fcf6c82b5/jdyyaac/obj/w5rDlsOJwrLDjj7CmsOj/22230141328/911d/4178/9503/23fcddca08575c4cda0327b79659b51d.m4a?authSecret=000001866cc099571cfe0aaba0900020',
        cover: 'https://s2.music.126.net/style/web2/img/frame/topbar.png?9eda0620c135d640f4d5d356b589735e',
    }, 
    {
        name: 'After LIKE',
        artist: 'IVE',
        url: 'https://m804.music.126.net/20230220112404/6eef5a560e03cd2ad2457cb2f8d9e61a/jdyyaac/obj/w5rDlsOJwrLDjj7CmsOj/23633884141/afa1/cd57/25e0/cd11d717fb252428f786a0006d6b98de.m4a?authSecret=000001866cc2b2f20af10aaba0ae67c2',
        cover: 'https://s2.music.126.net/style/web2/img/frame/topbar.png?9eda0620c135d640f4d5d356b589735e',
    },
    // {
    //     name: 'uptown funk',
    //     artist: 'Mark Ronson/Bruno Mars',
    //     url: 'https://m801.music.126.net/20221203234037/0f6d7004c92a3ff703bab9892c78e97c/jdyyaac/obj/w5rDlsOJwrLDjj7CmsOj/22259852841/1596/67c9/8826/66b034328865d1a40463683d1f345de5.m4a',
    //     cover: 'https://p2.music.126.net/fS_5RbP_4qg-RYreqp2tGg==/109951167558017839.jpg?param=130y130',
    // }, 
	]
});

