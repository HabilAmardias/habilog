export function formatDate(date, dateStyle, locales) {
	const dateToFormat = new Date(date.replaceAll('-', '/'))
	const dateFormatter = new Intl.DateTimeFormat(locales, { dateStyle })
	return dateFormatter.format(dateToFormat)
}

export function paginate(data) {
	const pages = [];
	for (let i = 0; i < data.length; i += 9) {
	  pages.push(data.slice(i, i + 9));
	}
	return pages;
  }