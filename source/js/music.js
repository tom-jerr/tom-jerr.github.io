const ap = new APlayer({
    container: document.getElementById('aplayer'),
    fixed: true,
	autoplay: true, //自动播放
    audio: [
	{
        name: 'uptown funk',
        artist: 'Mark Ronson/Bruno Mars',
        url: 'https://m801.music.126.net/20221203234037/0f6d7004c92a3ff703bab9892c78e97c/jdyyaac/obj/w5rDlsOJwrLDjj7CmsOj/22259852841/1596/67c9/8826/66b034328865d1a40463683d1f345de5.m4a',
        cover: 'https://p2.music.126.net/fS_5RbP_4qg-RYreqp2tGg==/109951167558017839.jpg?param=130y130',
    }, 
	]
});
